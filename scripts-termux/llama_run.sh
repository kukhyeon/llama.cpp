# this script should be run on llama.cpp/ dir.

# Hardware stats controls:
#   IGNITE_RECORD_MEMINFO=1|0
#   IGNITE_RECORD_COOLING=1|0
IGNITE_RECORD_MEMINFO="${IGNITE_RECORD_MEMINFO:-0}"
IGNITE_RECORD_COOLING="${IGNITE_RECORD_COOLING:-1}"
export IGNITE_RECORD_MEMINFO IGNITE_RECORD_COOLING

# screen brightness control
echo 0 > /sys/class/backlight/panel0-backlight/brightness

# silver core control
# su -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu3/online"

# CPU Governor: performance
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo "CPU Governor (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "CPU Governor (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
sleep 3

./build/bin/ignite \
    -m models/Llama-3.2-3B-Instruct-Q8_0.gguf \
    -i -cnv -tb 6 -t 6 -ub 512 -b 512 \
    -c 1024 \
    --temp 0 \
    --top-k 1 \
    --device-name S25 \
    --output-dir output/ \
    --input-path data/llama32_prefill_256.json \
    -fa off \
    --strict on \
    --strict-limit 128 \
    --max-query-number 5 \
    --cpu-p 15 \
    --ram-p 9 \
    --cpu-d 15 \
    --ram-d 9

# --layer-pause LP[ms]

# su -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu3/online"

# CPU Governor reset: walt
echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo "CPU Governor reset (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "CPU Governor reset (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"

# experiment done -> let screen brightness bright again
echo 1023 > /sys/class/backlight/panel0-backlight/brightness
