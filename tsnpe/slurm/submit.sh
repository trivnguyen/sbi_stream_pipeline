#!/usr/bin/env bash
# submit.sh — submit run_pipeline.sh to SLURM with per-run log bookkeeping.
#
# Usage:
#   ./submit.sh <config.py> [--rounds=N] [--start-round=R] [--time=D-HH:MM:SS] [--partition=name] [...] [--config.field=value ...]
#
# Examples:
#   ./submit.sh ../configs/draco.py --rounds=5
#   ./submit.sh ../configs/draco.py --rounds=5 --start-round=3
#   ./submit.sh ../configs/draco.py --rounds=5 --partition=debug --time=00:15:00
#
# Resource overrides — trailing --flag=value or env var, all optional
# (--config.<field>=<value> overrides go straight through to python):
#   --account=      / ACCOUNT      (default: def-tingli)
#   --partition=    / PARTITION    (default: compute)   # debug | compute | compute_h200 | *_full_node
#   --gpus=         / GPUS         (default: 1)         # keep at 1 - train_round.py isn't DDP-ready
#   --cpus=         / CPUS         (default: 24)        # full quarter-node cores; no mem flag, it's a no-op here
#   --time=         / TIME         (default: 12:00:00)  # compute partition max is 1 day
#   --rounds=       / ROUNDS       (default: 1)         # last round to run, inclusive
#   --start-round=  / START_ROUND  (default: 1)         # first round to run - bump this to
#                                                        # resume a partially-completed run
#                                                        # (already-done steps no-op regardless,
#                                                        # see run_pipeline.sh)
#
# $HOME is read-only on compute nodes, so all run artifacts (SLURM out/err,
# manifest, config snapshot, stdout log) live under $SCRATCH instead, one
# directory per submission: $SCRATCH/slurm_logs/tsnpe/<config_name>/<timestamp>/.
# A one-line-per-run index is also kept at $SCRATCH/slurm_logs/tsnpe/runs.tsv.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.py> [--rounds=N] [--start-round=R] [--config.field=value ...]" >&2
    exit 1
fi

CONFIG_ARG="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSNPE_DIR="$(dirname "$SCRIPT_DIR")"

# Resolve config to an absolute path (accept paths relative to cwd or to tsnpe/).
if [[ -f "$CONFIG_ARG" ]]; then
    CONFIG_ABS="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
elif [[ -f "$TSNPE_DIR/$CONFIG_ARG" ]]; then
    CONFIG_ABS="$(cd "$(dirname "$TSNPE_DIR/$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
else
    echo "Error: config not found: $CONFIG_ARG" >&2
    exit 1
fi

ACCOUNT="${ACCOUNT:-def-tingli}"
PARTITION="${PARTITION:-compute}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-24}"
TIME="${TIME:-12:00:00}"
ROUNDS="${ROUNDS:-1}"
START_ROUND="${START_ROUND:-1}"

# Trailing --flag=value overrides win over the env vars above; anything
# else (e.g. --config.field=value) passes through to run_pipeline.sh.
REMAINING_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --account=*)      ACCOUNT="${arg#*=}" ;;
        --partition=*)    PARTITION="${arg#*=}" ;;
        --gpus=*)         GPUS="${arg#*=}" ;;
        --cpus=*)         CPUS="${arg#*=}" ;;
        --time=*)         TIME="${arg#*=}" ;;
        --rounds=*)       ROUNDS="${arg#*=}" ;;
        --start-round=*)  START_ROUND="${arg#*=}" ;;
        *)                REMAINING_ARGS+=("$arg") ;;
    esac
done
EXTRA_ARGS=("${REMAINING_ARGS[@]}")

LOG_ROOT="${LOG_ROOT:-${SCRATCH:-/scratch/$(whoami)}/slurm_logs/tsnpe}"
CONFIG_NAME="$(basename "$CONFIG_ABS" .py)"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$LOG_ROOT/$CONFIG_NAME/$RUN_ID"
mkdir -p "$RUN_DIR"
cp -- "$CONFIG_ABS" "$RUN_DIR/config_snapshot.py"

{
    echo "submitted_at:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "submitted_by:    $(whoami)@$(hostname)"
    echo "config:          $CONFIG_ABS"
    echo "rounds:          $START_ROUND .. $ROUNDS"
    echo "extra_args:      ${EXTRA_ARGS[*]:-<none>}"
    echo "account:         $ACCOUNT"
    echo "partition:       $PARTITION"
    echo "gpus_per_node:   $GPUS"
    echo "cpus_per_task:   $CPUS"
    echo "time_limit:      $TIME"
} > "$RUN_DIR/manifest.txt"

JOB_NAME="tsnpe_${CONFIG_NAME}"

JOBID="$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node="$GPUS" \
    --cpus-per-task="$CPUS" \
    --time="$TIME" \
    --job-name="$JOB_NAME" \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",TSNPE_DIR="$TSNPE_DIR" \
    "$SCRIPT_DIR/run_pipeline.sbatch" "$CONFIG_ABS" --rounds "$ROUNDS" --start-round "$START_ROUND" "${EXTRA_ARGS[@]}")"

echo "job_id:          $JOBID" >> "$RUN_DIR/manifest.txt"

RUNS_INDEX="$LOG_ROOT/runs.tsv"
if [[ ! -f "$RUNS_INDEX" ]]; then
    printf 'job_id\tsubmitted_at\tconfig_name\tpartition\trounds\trun_dir\textra_args\n' > "$RUNS_INDEX"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOBID" "$(date '+%Y-%m-%d %H:%M:%S')" "$CONFIG_NAME" "$PARTITION" "$START_ROUND-$ROUNDS" "$RUN_DIR" "${EXTRA_ARGS[*]:-}" \
    >> "$RUNS_INDEX"

echo "Submitted job $JOBID ($JOB_NAME, partition=$PARTITION, gpu:$GPUS, rounds=$START_ROUND..$ROUNDS)"
echo "Logs: $RUN_DIR"
echo "Track: squeue -j $JOBID   |   tail -f $RUN_DIR/slurm-$JOBID.out"
