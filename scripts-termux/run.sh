# This script should be run on llama.cpp/ dir.

# Device detection (currently set to Pixel9, can be auto-detected)
# DEV="$(getprop ro.product.product.model)"
# DEV="$(printf '%s' "$DEV" | tr -d '[:space:]')"
DEV="S25"
echo "Device: $DEV"

# Turn-off screen (device-specific backlight paths)
if [ "$DEV" = "Pixel9" ]; then
  # Pixel9
  su -c "echo 0 > /sys/class/backlight/panel0-backlight/brightness"
elif [ "$DEV" = "S25" ]; then
  # S25
  su -c "echo 0 > /sys/class/backlight/panel0-backlight/brightness"
else
  # Default 
  su -c "echo 0 > /sys/class/backlight/panel/brightness"
fi

# CPU Governor: performance (for consistent performance during experiment)
if [ "$DEV" = "S25" ]; then
  su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
  su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"
  echo "CPU Governor (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
  echo "CPU Governor (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
fi
sleep 3

# Silver core control (Except S25)
if [ "$DEV" != "S25" ]; then
  su -c "echo 0 > /sys/devices/system/cpu/cpu1/online"
  su -c "echo 0 > /sys/devices/system/cpu/cpu2/online"
  su -c "echo 0 > /sys/devices/system/cpu/cpu3/online"
fi

# Run ignite with enhanced parameters
# CPU/RAM frequency parameters can be passed as script arguments
# Usage: ./run.sh [cpu_freq_index] [ram_freq_index]
CPU_FREQ=${1:-0}  # Default to 0 if not provided
RAM_FREQ=${2:-0}  # Default to 0 if not provided

./build/bin/ignite \
    -m ./models/qwen-1.5-0.5b-chat-q4_k_m.gguf \
    -cnv -i -tb 1 -t 4 -ub 512 -b 512 \
    --temp 0 \
    --top-k 1 \
    --device-name "$DEV" \
    --output-path output/hotpot_15_9.csv \
    --json-path dataset/hotpot_qa_30.json \
    --strict on \
    --strict-length 64 \
    --max-query-number 30 \
    --cpu-freq "$CPU_FREQ" \
    --ram-freq "$RAM_FREQ"

# Silver core reset (except S25)
if [ "$DEV" != "S25" ]; then
  su -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
  su -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
  su -c "echo 1 > /sys/devices/system/cpu/cpu3/online"
fi

# CPU Governor reset to default
if [ "$DEV" = "S25" ]; then
  su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
  su -c "echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"
  echo "CPU Governor reset (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
  echo "CPU Governor reset (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
fi

# Turn-on screen (device-specific brightness restoration)
if [ "$DEV" = "Pixel9" ]; then
  su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"
elif [ "$DEV" = "S25" ]; then
  su -c "echo 1023 > /sys/class/backlight/panel0-backlight/brightness"
else
  su -c "echo 1023 > /sys/class/backlight/panel/brightness"
fi

echo "Experiment completed for device: $DEV"

