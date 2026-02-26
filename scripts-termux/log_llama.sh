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
su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"

su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
su -c "echo $CLK0 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
su -c "chmod 444 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"

su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"
su -c "echo $CLK0 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"
su -c "chmod 444 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"

su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"
su -c "echo $CLK6 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"
su -c "chmod 444 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"

su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"
su -c "echo $CLK6 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"
su -c "chmod 444 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"

echo "[setup] CPU gov: $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor') / $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor')"
sleep 2

# ---- Output paths ----
mkdir -p output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACE_BASENAME="npu_llama_cpp_cpu_p0_12_p6_12_${TIMESTAMP}.perfetto-trace"
LOG_FILE="output/npu_llama_cpp_cpu_p0_12_p6_12_${TIMESTAMP}.csv"
TRACE_OUT="output/${TRACE_BASENAME}"

# perfetto는 /data/misc/perfetto-traces/ 에 먼저 저장 후 output/으로 복사
TRACE_TMP="/data/misc/perfetto-traces/${TRACE_BASENAME}"
TRACE_PUBLIC="/data/local/tmp/${TRACE_BASENAME}"

# config/pid 파일은 /data/local/tmp/ 에 (root 접근 가능)
PERFETTO_CFG_LOCAL="/data/local/tmp/perfetto_cfg_$$.pbtxt"
PERFETTO_PID_FILE="/data/local/tmp/perfetto_pid_$$.txt"
LOG_SCRIPT="/data/local/tmp/hw_log_cmd_$$.sh"

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

# ---- Write logging helper script ----
cat > "$LOG_SCRIPT" << 'EOF'
awk '{printf "%s,", $1/1000}' /sys/devices/virtual/thermal/thermal_zone*/temp
awk '{printf "%s,", $1}' /sys/class/kgsl/kgsl-3d0/devfreq/min_freq
awk '{printf "%s,", $1}' /sys/class/kgsl/kgsl-3d0/devfreq/max_freq
awk '{printf "%s,", $1/1000}' /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
awk '{printf "%s,", $1/1000}' /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
awk '{printf "%s,", $1/1000}' /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
awk '{printf "%s,", $1/1000}' /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq
awk '{printf "%s,", $2/1024}' /proc/meminfo
awk '{printf "%s,", $1}' /sys/class/power_supply/battery/power_now
awk '{printf "%s,", $1}' /sys/class/power_supply/battery/current_now
awk '{printf "%s,", $1}' /sys/class/power_supply/battery/voltage_now
awk '{printf "%s", $1/1000}' /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq
printf "\n"
EOF

# ---- Perfetto config ----
cat > "$PERFETTO_CFG_LOCAL" << 'EOF'
buffers { size_kb: 512000 fill_policy: RING_BUFFER }
data_sources {
  config {
    name: "linux.ftrace"
    ftrace_config {
      compact_sched { enabled: true }
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
      ftrace_events: "sched/sched_migrate_task"
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/cpu_idle"
      ftrace_events: "clk/clk_set_rate"
    }
  }
}
data_sources {
  config {
    name: "linux.sys_stats"
    sys_stats_config {
      meminfo_period_ms: 1000
      cpufreq_period_ms: 100
      devfreq_period_ms: 100
      psi_period_ms: 1000
    }
  }
}
data_sources {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 1000
      record_process_runtime: true
    }
  }
}
data_sources {
  config { name: "linux.system_info" }
}
EOF

su -c "mkdir -p /data/misc/perfetto-traces"
su -c "chmod 644 $PERFETTO_CFG_LOCAL"

# ---- Start Perfetto: root 쉘 안에서 백그라운드, PID 저장 ----
su -c "/system/bin/perfetto --txt -c $PERFETTO_CFG_LOCAL -o $TRACE_TMP >/data/local/tmp/perfetto_$$.log 2>&1 & echo \$! > $PERFETTO_PID_FILE"
sleep 0.5
echo "[perfetto] started (PID=$(su -c "cat $PERFETTO_PID_FILE" 2>/dev/null)), trace: $TRACE_TMP"

# ---- Start HW logging loop (background) ----
START_MS=$(date +%s%3N)

hw_log_loop() {
    while true; do
        NOW_MS=$(date +%s%3N)
        ELAPSED=$(awk "BEGIN {printf \"%.3f\", ($NOW_MS - $START_MS) / 1000.0}")
        ROW=$(su -c "sh $LOG_SCRIPT" 2>/dev/null)
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
        --device HTP0"

echo "[inference] done."

# ---- Stop logging ----
kill "$LOG_PID" 2>/dev/null
wait "$LOG_PID" 2>/dev/null

# ---- Stop perfetto (SIGINT for graceful flush) ----
PERF_PID=$(su -c "cat $PERFETTO_PID_FILE" 2>/dev/null)
if [ -n "$PERF_PID" ]; then
    su -c "kill -INT $PERF_PID" 2>/dev/null
    # flush 대기
    i=0
    while [ $i -lt 50 ]; do
        su -c "kill -0 $PERF_PID" 2>/dev/null || break
        sleep 0.1
        i=$((i+1))
    done
fi

# ---- Copy trace to output/ ----
su -c "chmod 644 $TRACE_TMP" 2>/dev/null || true
su -c "cp $TRACE_TMP $TRACE_PUBLIC && chmod 644 $TRACE_PUBLIC" 2>/dev/null || true
cp -f "$TRACE_PUBLIC" "$TRACE_OUT" 2>/dev/null || true

# ---- Cleanup ----
rm -f "$PERFETTO_CFG_LOCAL" "$LOG_SCRIPT"
su -c "rm -f $PERFETTO_PID_FILE /data/local/tmp/perfetto_$$.log" 2>/dev/null || true

echo "[result] HW log  : $LOG_FILE"
echo "[result] Perfetto: $TRACE_OUT"

# ---- Restore CPU freq ----
su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
su -c "echo 3532800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"
su -c "echo 384000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"

su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"
su -c "echo 4473600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"
su -c "chmod 644 /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"
su -c "echo 1017600 > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"

su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"

echo "[restore] CPU freq/governor restored"

# ---- Screen on ----
su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"

echo "[done]"