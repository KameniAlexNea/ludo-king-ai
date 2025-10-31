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
TOTAL_STEPS=${TOTAL_STEPS:-10000000}
N_STEPS=${N_STEPS:-512}
BATCH_SIZE=${BATCH_SIZE:-256}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
ENT_COEF=${ENT_COEF:-0.1}
VF_COEF=${VF_COEF:-0.5}
GAMMA=${GAMMA:-0.99}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}
SAVE_STEPS=${SAVE_STEPS:-400000}
DEVICE=${DEVICE:-cpu}
MAX_TURNS=${MAX_TURNS:-500}
OPPONENT_STRATEGY=${OPPONENT_STRATEGY:-probabilistic_v3}
FIXED_COLOR=${FIXED_COLOR:-RED}
PI_NET_ARCH=${PI_NET_ARCH:-"64 64"}
VF_NET_ARCH=${VF_NET_ARCH:-"256 256"}
N_ENVS=${N_ENVS:-16}
EVAL_FREQ=${EVAL_FREQ:-400000}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_OPPONENTS=${EVAL_OPPONENTS:-}
EVAL_DETERMINISTIC=${EVAL_DETERMINISTIC:-}

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
    --n-envs "${N_ENVS}"
    --eval-freq "${EVAL_FREQ}"
    --eval-episodes "${EVAL_EPISODES}"
)

if [[ -n "${FIXED_COLOR}" ]]; then
    TRAIN_ARGS+=(--fixed-agent-color "${FIXED_COLOR}")
fi

if [[ -n "${PI_NET_ARCH}" ]]; then
    # shellcheck disable=SC2206
    PI_NET_ARCH_VALUES=(${PI_NET_ARCH})
    TRAIN_ARGS+=(--pi-net-arch "${PI_NET_ARCH_VALUES[@]}")
fi

if [[ -n "${VF_NET_ARCH}" ]]; then
    # shellcheck disable=SC2206
    VF_NET_ARCH_VALUES=(${VF_NET_ARCH})
    TRAIN_ARGS+=(--vf-net-arch "${VF_NET_ARCH_VALUES[@]}")
fi

if [[ -n "${EVAL_OPPONENTS}" ]]; then
    TRAIN_ARGS+=(--eval-opponents "${EVAL_OPPONENTS}")
fi

if [[ -n "${EVAL_DETERMINISTIC}" ]]; then
    case "${EVAL_DETERMINISTIC,,}" in
        true|1|yes)
            ;; # deterministic is the default; no extra flag needed
        false|0|no)
            TRAIN_ARGS+=(--eval-stochastic)
            ;;
        *)
            echo "Warning: EVAL_DETERMINISTIC='${EVAL_DETERMINISTIC}' not understood; skipping." >&2
            ;;
    esac
fi

OUTPUT_FILE="${LOGDIR}/training.log"

echo "Launching training. Logs: ${OUTPUT_FILE}" >&2
echo "Command: python ${ROOT_DIR}/src/train.py ${TRAIN_ARGS[*]}" >&2

nohup python "${ROOT_DIR}/src/train.py" "${TRAIN_ARGS[@]}" \
    >>"${OUTPUT_FILE}" 2>&1 &

echo $! > "${LOGDIR}/training.pid"
echo "Training started with PID $(cat "${LOGDIR}/training.pid")" >&2