#!/bin/sh
# log_llama.sh - S25 NPU inference with HW logging + Perfetto
# Run from: /data/local/tmp/llama.cpp

# ---- Pin this shell to CPU0 ----
taskset -p 0x01 $$

# ---- S25 CPU freq (kHz) ----
CLK0=2918400
CLK6=3840000

echo "[setup] policy0=${CLK0} kHz, policy6=${CLK6} kHz"

# ---- Screen off ----
su -c "echo 0 > /sys/class/backlight/panel0-backlight/brightness"

# ---- CPU governor: performance + fix freq ----
su -c "
    echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
    echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor

    chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
    echo $CLK0 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
    chmod 444 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
    echo $CLK0 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
    chmod 444 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
    echo $CLK6 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
    chmod 444 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
    echo $CLK6 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
    chmod 444 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
"

echo "[setup] CPU gov: $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor') / $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor')"
sleep 2

# ---- Output paths ----
mkdir -p output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="output/npu_llama_cpp_cpu_p0_12_p6_12_${TIMESTAMP}.csv"
TRACE_FILE="output/npu_llama_cpp_cpu_p0_12_p6_12_${TIMESTAMP}.perfetto-trace"
PERFETTO_CFG="/tmp/perfetto_cfg_$$.pbtxt"

# ---- CSV header ----
THERMAL_NAMES=$(su -c "cat /sys/devices/virtual/thermal/thermal_zone*/type" 2>/dev/null | tr '\n' ',')
MEMINFO_NAMES=$(su -c "awk '{print \$1}' /proc/meminfo" 2>/dev/null | tr -d ':' | tr '\n' ',')

printf "Time,%s" "$THERMAL_NAMES" > "$LOG_FILE"
printf "gpu_min_clock,gpu_max_clock," >> "$LOG_FILE"
printf "cpu0_max_freq,cpu0_cur_freq,cpu6_max_freq,cpu6_cur_freq," >> "$LOG_FILE"
printf "%s" "$MEMINFO_NAMES" >> "$LOG_FILE"
printf "power_now,current_now,voltage_now," >> "$LOG_FILE"
printf "ddr_cur_freq\n" >> "$LOG_FILE"

echo "[log] CSV: $LOG_FILE"

# ---- Perfetto config ----
cat > "$PERFETTO_CFG" << 'EOF'
duration_ms: 7200000
buffers {
  size_kb: 131072
  fill_policy: RING_BUFFER
}
data_sources {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/cpu_idle"
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup_new"
      ftrace_events: "sched/sched_waking"
    }
  }
}
data_sources {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 1000
    }
  }
}
EOF

# ---- Start Perfetto (background) ----
su -c "/system/bin/perfetto --txt -c $PERFETTO_CFG -o $TRACE_FILE" &
PERFETTO_PID=$!
echo "[perfetto] started (PID=${PERFETTO_PID}), trace: $TRACE_FILE"

# ---- Start HW logging loop (background, CPU0 affinity inherited) ----
START_MS=$(date +%s%3N)

hw_log_loop() {
    while true; do
        NOW_MS=$(date +%s%3N)
        ELAPSED=$(awk "BEGIN {printf \"%.3f\", (${NOW_MS} - ${START_MS}) / 1000.0}")

        ROW=$(su -c "
            awk '{printf \"%s,\", \$1/1000}' /sys/devices/virtual/thermal/thermal_zone*/temp
            awk '{print \$1}' /sys/class/kgsl/kgsl-3d0/devfreq/min_freq | tr -d '\n'; printf ','
            awk '{print \$1}' /sys/class/kgsl/kgsl-3d0/devfreq/max_freq | tr -d '\n'; printf ','
            awk '{print \$1/1000}' /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq | tr -d '\n'; printf ','
            awk '{print \$1/1000}' /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq | tr -d '\n'; printf ','
            awk '{print \$1/1000}' /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq | tr -d '\n'; printf ','
            awk '{print \$1/1000}' /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq | tr -d '\n'; printf ','
            awk '{printf \"%s,\", \$2/1024}' /proc/meminfo
            awk '{print}' /sys/class/power_supply/battery/power_now | tr -d '\n'; printf ','
            awk '{print}' /sys/class/power_supply/battery/current_now | tr -d '\n'; printf ','
            awk '{print}' /sys/class/power_supply/battery/voltage_now | tr -d '\n'; printf ','
            awk '{print \$1/1000}' /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq
        " 2>/dev/null)

        printf "%s,%s\n" "$ELAPSED" "$ROW" >> "$LOG_FILE"
        sleep 0.3
    done
}

hw_log_loop &
LOG_PID=$!
echo "[log] HW logging started (PID=${LOG_PID})"

# ---- Run NPU inference (CPU 4-7 via taskset f0) ----
echo "[inference] starting..."
su -p -c "taskset f0 setenforce 0 && \
    export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib && \
    export ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib && \
    export GGML_HEXAGON_HOSTBUF=1 && \
    cd /data/local/tmp/llama.cpp && \
    taskset f0 ./bin/llama-ignite-npu \
        -m /data/local/tmp/gguf/qwen1_5-0_5b-chat-q4_k_m.gguf \
        -t 1 -np 1 -ub 512 -b 512 \
        --json-path data/hotpot_qa_20.json \
        --output-dir output \
        --strict 1 \
        --strict-limit 64 \
        -c 1024 \
        --cpu-mask 0xfc \
        --device HTP0"

echo "[inference] done."

# ---- Stop logging and perfetto ----
kill "$LOG_PID" 2>/dev/null
kill "$PERFETTO_PID" 2>/dev/null
wait "$LOG_PID" 2>/dev/null
wait "$PERFETTO_PID" 2>/dev/null

rm -f "$PERFETTO_CFG"

echo "[result] HW log  : $LOG_FILE"
echo "[result] Perfetto: $TRACE_FILE"

# ---- Restore CPU freq ----
su -c "
    chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
    echo 3532800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
    echo 384000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
    echo 4473600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq

    chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq
    echo 1017600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq

    echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
    echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
"
echo "[restore] CPU freq/governor restored"

# ---- Screen on ----
su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"

echo "[done]"