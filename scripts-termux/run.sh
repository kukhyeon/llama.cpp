# this script should be run on llama.cpp/ dir.

# screen brightness control
su -c "echo 0 > /sys/class/backlight/panel0-backlight/brightness"

# silver core control
# su -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu3/online"

# CPU Governor: performance
su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"
echo "CPU Governor (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "CPU Governor (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
sleep 3

./build/bin/ignite \
    -m ~/.cache/llama.cpp/tensorblock_Qwen1.5-0.5B-GGUF_Qwen1.5-0.5B-Q4_K.gguf \ 
    -i -cnv -tb 1 -t 4 -ub 512 -b 512 \
    -c 1024 \
    --temp 0 \
    --top-k 1 \
    --device-name S25 \
    --output-dir outputs/ \
    --json-path dataset/hotpot_qa_30.json \
    --strict on \
    --strict-length 64 \
    --max-query-number 30 \
    --cpu-p 15 \
    --ram-d 9 \
    --cpu-p 15 \
    --ram-d 9

# --layer-pause LP[ms]

# su -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
# su -c "echo 1 > /sys/devices/system/cpu/cpu3/online"

# CPU Governor reset: walt
su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"
echo "CPU Governor reset (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "CPU Governor reset (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"

# experiment done -> let screen brightness bright again
su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"

