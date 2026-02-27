#!/bin/sh
# log_llama.sh - S25 NPU inference with HW logging (thermal/clock/power/CPU util)
# Run from: /data/local/tmp/llama.cpp
# Usage: su -c "sh log_llama.sh"

# ---- Pin this shell to CPU0 ----
taskset -p 0x01 $$


# ---- Cleanup on exit (normal/SIGINT/SIGTERM) ----
cleanup() {
    kill "$LOG_PID" 2>/dev/null
    wait "$LOG_PID" 2>/dev/null
    # su 자식 프로세스(sh hw_log_cmd, awk 등)는 root 소유 -> su -c pkill로 정리
    su -c "pkill -f hw_log_cmd" 2>/dev/null
    su -c "pkill -f 'awk.*thermal'" 2>/dev/null
    rm -f "$LOG_SCRIPT" 2>/dev/null
}
trap cleanup EXIT INT TERM
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
su -c "chmod 644 /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
su -c "echo 4761000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
su -c "chmod 444 /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
echo "[setup] DDR boost_freq: $(su -c 'cat /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq')"

echo "[setup] CPU gov: $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor') / $(su -c 'cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor')"
sleep 2

# ---- Output paths ----
mkdir -p output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="output/npu_llama_cpp_cpu_p0_12_p6_12_${TIMESTAMP}.csv"
LOG_SCRIPT="/data/local/tmp/hw_log_cmd_$$.sh"

# ---- CSV header ----
THERMAL_NAMES=$(su -c "cat /sys/devices/virtual/thermal/thermal_zone*/type" 2>/dev/null | tr '\n' ',')
MEMINFO_NAMES=$(su -c "awk '{print \$1}' /proc/meminfo" 2>/dev/null | tr -d ':' | tr '\n' ',')

printf "Time,%s" "$THERMAL_NAMES" > "$LOG_FILE"
printf "gpu_min_clock,gpu_max_clock," >> "$LOG_FILE"
printf "cpu0_max_freq,cpu0_cur_freq,cpu6_max_freq,cpu6_cur_freq," >> "$LOG_FILE"
printf "%s" "$MEMINFO_NAMES" >> "$LOG_FILE"
printf "power_now,current_now,voltage_now," >> "$LOG_FILE"
printf "ddr_cur_freq," >> "$LOG_FILE"
printf "cpu0_util,cpu1_util,cpu2_util,cpu3_util,cpu4_util,cpu5_util,cpu6_util,cpu7_util\n" >> "$LOG_FILE"

echo "[log] CSV: $LOG_FILE"

# ---- Write logging helper script (sysfs reads, run as root) ----
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
awk '{printf "%s,", $1/1000}' /sys/devices/system/cpu/bus_dcvs/DDR/cur_freq
EOF

# ---- HW logging loop (background, CPU util from /proc/stat - no su needed) ----
START_MS=$(date +%s%3N)

hw_log_loop() {
    PREV_STAT=$(grep '^cpu[0-9]' /proc/stat)
    while true; do
        NOW_MS=$(date +%s%3N)
        ELAPSED=$(awk "BEGIN {printf \"%.3f\", ($NOW_MS - $START_MS) / 1000.0}")

        ROW=$(su -c "sh $LOG_SCRIPT" 2>/dev/null)

        CURR_STAT=$(grep '^cpu[0-9]' /proc/stat)
        CPU_UTIL=$(printf "%s\n%s" "$PREV_STAT" "$CURR_STAT" | awk '
        {
            core=$1; idle=$5; total=0
            for(i=2;i<=NF;i++) total+=$i
            if(seen[core]) {
                d_idle  = idle - prev_idle[core]
                d_total = total - prev_total[core]
                util = (d_total > 0) ? (d_total - d_idle) / d_total * 100 : 0
                printf "%.1f,", util
            }
            prev_idle[core]=idle; prev_total[core]=total; seen[core]=1
        }')
        PREV_STAT=$CURR_STAT

        printf "%s,%s%s\n" "$ELAPSED" "$ROW" "$CPU_UTIL" >> "$LOG_FILE"
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


echo "[result] HW log: $LOG_FILE"

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
su -c "chmod 644 /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
su -c "echo 5470000 > /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
su -c "chmod 444 /sys/devices/system/cpu/bus_dcvs/DDR/boost_freq"
echo "[restore] DDR boost_freq restored"
echo "[restore] CPU freq/governor restored"

# ---- Screen on ----
su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"

echo "[done]"