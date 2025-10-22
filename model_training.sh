#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script requires bash. Please run it with 'bash model_training.sh'." >&2
    exit 1
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

LOGDIR=${LOGDIR:-training/logs}
MODELDIR=${MODELDIR:-training/models}
TOTAL_STEPS=${TOTAL_STEPS:-5000000}
N_STEPS=${N_STEPS:-1024}
BATCH_SIZE=${BATCH_SIZE:-256}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
ENT_COEF=${ENT_COEF:-0.01}
VF_COEF=${VF_COEF:-0.5}
GAMMA=${GAMMA:-0.99}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}
SAVE_STEPS=${SAVE_STEPS:-400000}
DEVICE=${DEVICE:-cpu}
MAX_TURNS=${MAX_TURNS:-500}
OPPONENT_STRATEGY=${OPPONENT_STRATEGY:-probabilistic_v3}
FIXED_COLOR=${FIXED_COLOR:-}
NET_ARCH=${NET_ARCH:-"512 256 128"}

mkdir -p "${LOGDIR}" "${MODELDIR}"

TRAIN_ARGS=(
    --total-steps "${TOTAL_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --n-steps "${N_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --ent-coef "${ENT_COEF}"
    --vf-coef "${VF_COEF}"
    --gamma "${GAMMA}"
    --gae-lambda "${GAE_LAMBDA}"
    --logdir "${LOGDIR}"
    --model-dir "${MODELDIR}"
    --device "${DEVICE}"
    --save-steps "${SAVE_STEPS}"
    --max-turns "${MAX_TURNS}"
    --opponent-strategy "${OPPONENT_STRATEGY}"
)

if [[ -n "${FIXED_COLOR}" ]]; then
    TRAIN_ARGS+=(--fixed-agent-color "${FIXED_COLOR}")
fi

if [[ -n "${NET_ARCH}" ]]; then
    # shellcheck disable=SC2206
    NET_ARCH_VALUES=(${NET_ARCH})
    TRAIN_ARGS+=(--net-arch "${NET_ARCH_VALUES[@]}")
fi

OUTPUT_FILE="${LOGDIR}/training.log"

echo "Launching training. Logs: ${OUTPUT_FILE}" >&2

nohup python "${ROOT_DIR}/src/train.py" "${TRAIN_ARGS[@]}" \
    >>"${OUTPUT_FILE}" 2>&1 &

echo $! > "${LOGDIR}/training.pid"
echo "Training started with PID $(cat "${LOGDIR}/training.pid")" >&2