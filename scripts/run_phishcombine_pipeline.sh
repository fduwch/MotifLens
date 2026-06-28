#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON_BIN:-python}
DATASET=${1:-Meta}
HOPS=${2:-hop1}
EPOCHS=${3:-30}
ROOT=${PHISH_ROOT:-data/raw/PhishCombine}
BASE_DIR=${BASE_DIR:-$(pwd)}
MAX_ROWS_PER_FILE=${MAX_ROWS_PER_FILE:-0}
MAX_FILE_MB=${MAX_FILE_MB:-0}
MAX_EDGES_PER_PAIR=${MAX_EDGES_PER_PAIR:-0}
PAIR_CAP_SCOPE=${PAIR_CAP_SCOPE:-pair_tx_type_hop}
HIDDEN=${HIDDEN:-64}
HEADS=${HEADS:-2}
R_WEIGHT=${R_WEIGHT:-0.05}
MODEL=${MODEL:-sage}
TRAIN_MODE=${TRAIN_MODE:-full}
if [[ -z "${LR:-}" ]]; then
  if [[ "$MODEL" == "sage" || "$MODEL" == "gcn" ]]; then
    LR=1e-3
  else
    LR=5e-4
  fi
fi
LAYERS=${LAYERS:-2}
USE_EVIDENCE_FEATURES=${USE_EVIDENCE_FEATURES:-1}
EVIDENCE_TEACHER=${EVIDENCE_TEACHER:-rule}
LLM_BASE_URL=${LLM_BASE_URL:-}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3-14B}
LLM_API_KEY=${LLM_API_KEY:-EMPTY}
LLM_BATCH_SIZE=${LLM_BATCH_SIZE:-2}
LLM_MAX_CARDS=${LLM_MAX_CARDS:-0}
LLM_MAX_EDGES_PER_CARD=${LLM_MAX_EDGES_PER_CARD:-12}
LLM_MIN_CONFIDENCE=${LLM_MIN_CONFIDENCE:-0.35}
LLM_PROMPT_NO_THINK=${LLM_PROMPT_NO_THINK:-0}
LOSS=${LOSS:-ce}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
FANOUTS=${FANOUTS:-10,5}
BATCH_SIZE=${BATCH_SIZE:-512}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
if [[ -z "${PATIENCE:-}" ]]; then
  if [[ "$MODEL" == "sage" || "$MODEL" == "gcn" ]]; then
    PATIENCE=35
  else
    PATIENCE=20
  fi
fi

cd "$BASE_DIR"

DATA_DIR="data/phishcombine/${DATASET}_${HOPS}"
OUT_DIR="outputs/phishcombine/${DATASET}_${HOPS}"
mkdir -p "$OUT_DIR"

TWO_HOP_ARGS=()
if [[ "$HOPS" == "hop2" || "$HOPS" == "twohop" ]]; then
  TWO_HOP_ARGS=(--include-two-hop)
fi

LIMIT_ARGS=()
if [[ "$MAX_ROWS_PER_FILE" != "0" ]]; then
  LIMIT_ARGS+=(--max-rows-per-file "$MAX_ROWS_PER_FILE")
fi
if [[ "$MAX_FILE_MB" != "0" ]]; then
  LIMIT_ARGS+=(--max-file-mb "$MAX_FILE_MB")
fi
if [[ "$MAX_EDGES_PER_PAIR" != "0" ]]; then
  LIMIT_ARGS+=(--max-edges-per-pair "$MAX_EDGES_PER_PAIR" --pair-cap-scope "$PAIR_CAP_SCOPE")
fi

echo "== convert ${DATASET} ${HOPS} =="
"$PY" scripts/convert_phishcombine.py \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --out-dir "$DATA_DIR" \
  --include-meta-features \
  --progress-every 500 \
  "${TWO_HOP_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

EVIDENCE_PATH="$OUT_DIR/evidence.csv"
echo "== evidence ${DATASET} ${HOPS} teacher=${EVIDENCE_TEACHER} =="
if [[ "$EVIDENCE_TEACHER" == "rule" ]]; then
  "$PY" scripts/generate_evidence.py \
    --data-dir "$DATA_DIR" \
    --out "$EVIDENCE_PATH" \
    --teacher rule \
    --top-edges-per-source 20
elif [[ "$EVIDENCE_TEACHER" == "motif_llm" || "$EVIDENCE_TEACHER" == "llm_motif" ]]; then
  LLM_ARGS=(
    --data-dir "$DATA_DIR"
    --out "$OUT_DIR/llm_motif_evidence.csv"
    --cards-out "$OUT_DIR/observation_cards.jsonl"
    --responses-out "$OUT_DIR/llm_motif_responses.jsonl"
    --teacher llm
    --llm-base-url "$LLM_BASE_URL"
    --llm-model "$LLM_MODEL"
    --llm-api-key "$LLM_API_KEY"
    --llm-batch-size "$LLM_BATCH_SIZE"
    --max-cards "$LLM_MAX_CARDS"
    --max-edges-per-card "$LLM_MAX_EDGES_PER_CARD"
    --min-confidence "$LLM_MIN_CONFIDENCE"
  )
  if [[ "$LLM_PROMPT_NO_THINK" == "1" ]]; then
    LLM_ARGS+=(--prompt-no-think)
  fi
  "$PY" scripts/generate_llm_motif_evidence.py "${LLM_ARGS[@]}"
  EVIDENCE_PATH="$OUT_DIR/llm_motif_evidence.csv"
elif [[ "$EVIDENCE_TEACHER" == "motif_heuristic" ]]; then
  "$PY" scripts/generate_llm_motif_evidence.py \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR/llm_motif_evidence.csv" \
    --cards-out "$OUT_DIR/observation_cards.jsonl" \
    --responses-out "$OUT_DIR/llm_motif_responses.jsonl" \
    --teacher heuristic \
    --max-cards "$LLM_MAX_CARDS" \
    --max-edges-per-card "$LLM_MAX_EDGES_PER_CARD" \
    --min-confidence "$LLM_MIN_CONFIDENCE"
  EVIDENCE_PATH="$OUT_DIR/llm_motif_evidence.csv"
else
  echo "Unknown EVIDENCE_TEACHER=$EVIDENCE_TEACHER" >&2
  exit 2
fi

echo "== train ${DATASET} ${HOPS} =="
if [[ "$TRAIN_MODE" == "sampled" ]]; then
  if [[ "$MODEL" != "sage" ]]; then
    echo "TRAIN_MODE=sampled currently supports MODEL=sage only" >&2
    exit 2
  fi
  TRAIN_ARGS=(
    --data-dir "$DATA_DIR"
    --evidence "$EVIDENCE_PATH"
    --out-dir "$OUT_DIR/${MODEL}_sampled"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --hidden "$HIDDEN"
    --layers "$LAYERS"
    --fanouts "$FANOUTS"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --lr "$LR"
    --selection-metric val_best_f1
    --loss "$LOSS"
    --focal-gamma "$FOCAL_GAMMA"
    --label-smoothing "$LABEL_SMOOTHING"
    --grad-clip 1.0
  )
  if [[ "$USE_EVIDENCE_FEATURES" == "1" ]]; then
    TRAIN_ARGS+=(--use-evidence-features)
  fi
  "$PY" scripts/train_sampled.py "${TRAIN_ARGS[@]}"
else
  TRAIN_ARGS=(
    --data-dir "$DATA_DIR"
    --evidence "$EVIDENCE_PATH"
    --out-dir "$OUT_DIR/$MODEL"
    --model "$MODEL"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --hidden "$HIDDEN"
    --layers "$LAYERS"
    --lr "$LR"
    --rationale-weight "$R_WEIGHT"
    --add-reverse-edges
    --add-self-loops
    --selection-metric val_best_f1
    --loss "$LOSS"
    --focal-gamma "$FOCAL_GAMMA"
    --label-smoothing "$LABEL_SMOOTHING"
    --grad-clip 1.0
  )

  if [[ "$MODEL" == "rationale_gat" ]]; then
    TRAIN_ARGS+=(--heads "$HEADS")
  else
    TRAIN_ARGS+=(--heads 1 --rationale-weight 0)
  fi

  if [[ "$USE_EVIDENCE_FEATURES" == "1" ]]; then
    TRAIN_ARGS+=(--use-evidence-features)
  fi

  "$PY" scripts/train.py "${TRAIN_ARGS[@]}"
fi

echo "== done =="
if [[ "$TRAIN_MODE" == "sampled" ]]; then
  echo "$OUT_DIR/${MODEL}_sampled/metrics.json"
else
  echo "$OUT_DIR/$MODEL/metrics.json"
fi
