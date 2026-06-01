#!/bin/sh
# run.sh - NPU inference with runtime DVFS controlled by llama-ignite-npu
# Run from: /data/local/tmp/llama.cpp
#
# Optional positional args:
#   $1: prefill CPU DVFS index
#   $2: prefill RAM DVFS index
#   $3: decode CPU DVFS index
#   $4: decode RAM DVFS index
#   $5: phase pause in ms
#   $6: token pause in ms
#   $7: layer pause in ms
#   $8: ignite verbose [on|off]
#   $9: prefill GPU DVFS index
#   $10: decode GPU DVFS index
#
# Runtime option env controls:
#   THREADS=6
#   THREADS_BATCH=6
#   JSON_PATH=data/qwen3_prefill_64.json
#   MAX_QUERY_NUMBER=20
#   STRICT=on
#   STRICT_LIMIT=128
#   DEVICE=HTP0
#   GPU_P=-1
#   GPU_D=-1
#
# Backend policy env controls:
#   BACKEND_POLICY=on|off
#   BACKEND_POLICY_CONFIG=policy/qwen3.json
#   BACKEND_POLICY_WEIGHTS=on|off
#   BACKEND_POLICY_OPS=on|off
#
# Stage backend switch controls:
#   STAGE_SWITCH=on|off
#
# Thermal backend switch controls:
#   THERMAL_SWITCH=on|off
#   THERMAL_STATE_FILE=/data/local/tmp/llama.cpp/thermal_state.txt
#   THERMAL_PRIME_CPU=6
#   THERMAL_TOLERANCE_KHZ=0
#   THERMAL_DEBOUNCE=1
#
# CSV op load controls:
#   CSV_OP_LOAD=on|off
#   CSV_OP_LOAD_OPS=MUL_MAT,MUL_MAT_ID,CONT,SOFT_MAX,SET_ROWS

DEV="${DEV:-S25}"
MODEL="${MODEL:-/data/local/tmp/gguf/qwen-3-1.7b-q4_0.gguf}"
CPU_P="${1:-15}"
RAM_P="${2:-9}"
CPU_D="${3:-15}"
RAM_D="${4:-9}"
PHASE_PAUSE_MS="${5:-0}"
TOKEN_PAUSE_MS="${6:-0}"
LAYER_PAUSE_MS="${7:-0}"
IGNITE_VERBOSE="${8:-off}"
GPU_P="${GPU_P:-${9:--1}}"
GPU_D="${GPU_D:-${10:--1}}"
THREADS="${THREADS:-6}"
THREADS_BATCH="${THREADS_BATCH:-6}"
JSON_PATH="${JSON_PATH:-data/qwen3_prefill_64.json}"
MAX_QUERY_NUMBER="${MAX_QUERY_NUMBER:-20}"
STRICT="${STRICT:-on}"
STRICT_LIMIT="${STRICT_LIMIT:-128}"
DEVICE="${DEVICE:-HTP0}"
BACKEND_POLICY="${BACKEND_POLICY:-on}"
BACKEND_POLICY_WEIGHTS="${BACKEND_POLICY_WEIGHTS:-on}"
BACKEND_POLICY_OPS="${BACKEND_POLICY_OPS:-on}"
STAGE_SWITCH="${STAGE_SWITCH:-off}"
STAGE_POLICY_CONFIG="${STAGE_POLICY_CONFIG:-policy/qwen3_stage_switch.json}"
THERMAL_SWITCH="${THERMAL_SWITCH:-off}"
THERMAL_POLICY_CONFIG="${THERMAL_POLICY_CONFIG:-policy/qwen3_thermal_switch.json}"
THERMAL_STATE_FILE="${THERMAL_STATE_FILE:-/data/local/tmp/llama.cpp/thermal_state.txt}"
THERMAL_PRIME_CPU="${THERMAL_PRIME_CPU:-6}"
THERMAL_TOLERANCE_KHZ="${THERMAL_TOLERANCE_KHZ:-0}"
THERMAL_DEBOUNCE="${THERMAL_DEBOUNCE:-1}"
THERMAL_VERBOSE="${THERMAL_VERBOSE:-1}"
CSV_OP_LOAD="${CSV_OP_LOAD:-off}"
CSV_OP_LOAD_OPS="${CSV_OP_LOAD_OPS:-MUL_MAT,MUL_MAT_ID,CONT,SOFT_MAX,SET_ROWS}"
export CSV_OP_LOAD CSV_OP_LOAD_OPS

case "$STAGE_SWITCH" in
    1|on|ON|true|TRUE|yes|YES)
        STAGE_SWITCH_ON=1
        ;;
    *)
        STAGE_SWITCH_ON=0
        ;;
esac

case "$THERMAL_SWITCH" in
    1|on|ON|true|TRUE|yes|YES)
        THERMAL_SWITCH_ON=1
        ;;
    *)
        THERMAL_SWITCH_ON=0
        ;;
esac

if [ -z "${BACKEND_POLICY_CONFIG+x}" ]; then
    if [ "$THERMAL_SWITCH_ON" = "1" ]; then
        BACKEND_POLICY_CONFIG="$THERMAL_POLICY_CONFIG"
    elif [ "$STAGE_SWITCH_ON" = "1" ]; then
        BACKEND_POLICY_CONFIG="$STAGE_POLICY_CONFIG"
    else
        BACKEND_POLICY_CONFIG="policy/qwen3.json"
    fi
fi

if [ "$THERMAL_SWITCH_ON" = "1" ] && [ "$STAGE_SWITCH_ON" = "1" ]; then
    echo "[setup] Both THERMAL_SWITCH and STAGE_SWITCH are on; thermal policy config takes precedence."
    STAGE_SWITCH_ON=0
fi

case "$IGNITE_VERBOSE" in
    1|on|ON|true|TRUE|yes|YES)
        IGNITE_VERBOSE_ARG="--ignite-verbose"
        ;;
	    *)
	        IGNITE_VERBOSE_ARG=""
	        ;;
esac

case "$BACKEND_POLICY" in
    1|on|ON|true|TRUE|yes|YES)
        BACKEND_POLICY_ARGS="--backend-policy ${BACKEND_POLICY_CONFIG} --backend-policy-weights ${BACKEND_POLICY_WEIGHTS} --backend-policy-ops ${BACKEND_POLICY_OPS}"
        ;;
    *)
        BACKEND_POLICY_ARGS=""
        ;;
esac

restore_system_state() {
    status=$?

    echo 1023 > /sys/class/backlight/panel0-backlight/brightness 2>/dev/null || true
    echo "[restore] screen on"

    echo walt > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null || true
    echo walt > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor 2>/dev/null || true
    echo "CPU Governor reset (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null)"
    echo "CPU Governor reset (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor 2>/dev/null)"

    echo "[inference] done."

    trap - EXIT INT TERM
    exit "$status"
}

trap restore_system_state EXIT INT TERM

# screen brightness control
echo 0 > /sys/class/backlight/panel0-backlight/brightness

# Keep governor setup in the script, but let ignite-npu control per-phase DVFS.
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
echo "CPU Governor (policy0): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "CPU Governor (policy6): $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
sleep 2

echo "[setup] DVFS device: $DEV"
echo "[setup] DVFS indices: prefill(cpu=$CPU_P, ram=$RAM_P, gpu=$GPU_P), decode(cpu=$CPU_D, ram=$RAM_D, gpu=$GPU_D)"
echo "[setup] Phase pause: ${PHASE_PAUSE_MS}ms"
echo "[setup] Token pause: ${TOKEN_PAUSE_MS}ms"
echo "[setup] Layer pause: ${LAYER_PAUSE_MS}ms"
echo "[setup] Ignite verbose: ${IGNITE_VERBOSE}"
echo "[setup] Runtime: threads=${THREADS}, threads_batch=${THREADS_BATCH}, json=${JSON_PATH}, max_query=${MAX_QUERY_NUMBER}, strict=${STRICT}, strict_limit=${STRICT_LIMIT}, device=${DEVICE}"
echo "[setup] Backend policy: ${BACKEND_POLICY} (${BACKEND_POLICY_CONFIG}, weights=${BACKEND_POLICY_WEIGHTS}, ops=${BACKEND_POLICY_OPS})"
if [ "$STAGE_SWITCH_ON" = "1" ]; then
    echo "[setup] Stage switch: on"
fi
if [ "$THERMAL_SWITCH_ON" = "1" ]; then
    echo "[setup] Thermal switch: on (state=${THERMAL_STATE_FILE}, prime_cpu=${THERMAL_PRIME_CPU})"
fi

setenforce 0 || true

export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib
export ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib
export GGML_HEXAGON_HOSTBUF=1
export IGNITE_CSV_OP_BREAKDOWN=0

export LLAMA_BACKEND_POLICY_RESIDENCY=0
export LLAMA_BACKEND_POLICY_STAGE_SWITCH=0
export LLAMA_BACKEND_POLICY_THERMAL_SWITCH=0
export IGNITE_THERMAL_SIGNAL=0

if [ "$STAGE_SWITCH_ON" = "1" ]; then
    export LLAMA_BACKEND_POLICY_RESIDENCY=1
    export LLAMA_BACKEND_POLICY_STAGE_SWITCH=1
fi

if [ "$THERMAL_SWITCH_ON" = "1" ]; then
    export LLAMA_BACKEND_POLICY_RESIDENCY=1
    export LLAMA_BACKEND_POLICY_THERMAL_SWITCH=1
    export LLAMA_BACKEND_POLICY_THERMAL_STATE_FILE="$THERMAL_STATE_FILE"

    export IGNITE_THERMAL_SIGNAL=1
    export IGNITE_THERMAL_SIGNAL_FILE="$THERMAL_STATE_FILE"
    export IGNITE_THERMAL_PRIME_CPU="$THERMAL_PRIME_CPU"
    export IGNITE_THERMAL_TOLERANCE_KHZ="$THERMAL_TOLERANCE_KHZ"
    export IGNITE_THERMAL_DEBOUNCE="$THERMAL_DEBOUNCE"
    export IGNITE_THERMAL_VERBOSE="$THERMAL_VERBOSE"

    mkdir -p "$(dirname "$THERMAL_STATE_FILE")" 2>/dev/null || true
    echo "cool" > "$THERMAL_STATE_FILE" 2>/dev/null || true

    echo "[thermal-switch] trigger: scaling_cur_freq drops by more than ${THERMAL_TOLERANCE_KHZ}kHz for ${THERMAL_DEBOUNCE} sample(s)"
fi

cd /data/local/tmp/llama.cpp || exit 1

taskset fe ./bin/llama-ignite-npu \
    -m "$MODEL" \
    -t "$THREADS" -tb "$THREADS_BATCH" -i -cnv -ub 512 -b 512 -fa off \
    --json-path "$JSON_PATH" \
    --max-query-number "$MAX_QUERY_NUMBER" \
    --strict "$STRICT" \
    --strict-limit "$STRICT_LIMIT" \
    --output-dir output \
    --temp 0 \
    --top-k 1 \
    -c 1024 \
    --device "$DEVICE" \
    --backend-compute-profile \
    --dvfs-device "$DEV" \
    --cpu-p "$CPU_P" \
    --ram-p "$RAM_P" \
    --gpu-p "$GPU_P" \
    --cpu-d "$CPU_D" \
    --ram-d "$RAM_D" \
    --gpu-d "$GPU_D" \
    --phase-pause "$PHASE_PAUSE_MS" \
	--token-pause "$TOKEN_PAUSE_MS" \
	--layer-pause "$LAYER_PAUSE_MS" \
    ${BACKEND_POLICY_ARGS} \
	${IGNITE_VERBOSE_ARG}
