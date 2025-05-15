#!/usr/bin/env bash
set -euo pipefail

cd /oscar/data/yma16/Project/Cross_species/scripts_STAGATE_ST_VAE_NB_nonHomo

# ─── COMMON PATHS & PARAMS ────────────────────────────────────────────────────
REF_PATH="/oscar/data/yma16/Project/Cross_species/00_Data/Slides-seqV2_kidney/healthy_human/Puck_200104_15.h5ad"
TARGET_PATH="/oscar/data/yma16/Project/Cross_species/00_Data/Slides-seqV2_kidney/healthy_mouse/Puck_191109_07.h5ad"
SAVEDIR="/oscar/data/yma16/Project/Cross_species/01_results_NBVAE_Slide-seqV2"

BASE_NAME="VAE_NB_nonhomo"
SPECIES1="Human"
SPECIES2="Mouse"
BETA_KL="0.01"
EPOCHS="80"
LR="2e-4"

# ─── HYPERPARAM COMBO (only one right now) ───────────────────────────────────
# alpha beta_cls beta_mmd beta_coral
COMBOS=(
  "1 1 1 0"
)

# ─── FLAG CONFIGURATIONS IN ORDER ─────────────────────────────────────────────
FLAG_CONFIGS=(
  # VAE + condition + denoise + identity_graph + preprocess
  'VAE_FLAG="--VAE"    CONDITION_FLAG="--condition" DENOISE_FLAG="--denoise"    IDENTITY_GRAPH_FLAG="--identity_graph" PREPROCESS_FLAG=""'
  # # VAE + condition + no denoise + identity_graph
  # 'VAE_FLAG="--VAE"    CONDITION_FLAG="--condition" DENOISE_FLAG=""             IDENTITY_GRAPH_FLAG="--identity_graph" PREPROCESS_FLAG=""'
  # # VAE + no condition + denoise + identity_graph
  # 'VAE_FLAG="--VAE"    CONDITION_FLAG=""            DENOISE_FLAG="--denoise"    IDENTITY_GRAPH_FLAG="--identity_graph" PREPROCESS_FLAG=""'
  # # VAE + no condition + no denoise + identity_graph
  # 'VAE_FLAG="--VAE"    CONDITION_FLAG=""            DENOISE_FLAG=""             IDENTITY_GRAPH_FLAG="--identity_graph" PREPROCESS_FLAG=""'
)

# ─── LOOP OVER COMBOS & FLAG SETS ───────────────────────────────────────────────
for combo in "${COMBOS[@]}"; do
  read -r ALPHA BETA_CLS BETA_MMD BETA_CORAL <<< "$combo"

  for cfg in "${FLAG_CONFIGS[@]}"; do
    # apply this set of flags
    eval "$cfg"

    # build dynamic name from flags
    name_parts=()
    [[ -n $CONDITION_FLAG ]] && name_parts+=("Condition")
    [[ -n $DENOISE_FLAG   ]] && name_parts+=("Denoise")
    name_parts+=("$BASE_NAME")
    NAME=$(IFS=_; echo "${name_parts[*]}")

    echo "➡️ Running: ALPHA=${ALPHA}, BETA_CLS=${BETA_CLS}, BETA_MMD=${BETA_MMD}, BETA_CORAL=${BETA_CORAL}"
    echo "   flags: ${VAE_FLAG} ${CONDITION_FLAG} ${DENOISE_FLAG} ${IDENTITY_GRAPH_FLAG} ${PREPROCESS_FLAG}"
    echo "   name:  ${NAME}"

    python main_run.py \
      --ref_path        "$REF_PATH" \
      --target_path     "$TARGET_PATH" \
      --name            "$NAME" \
      --savedir         "$SAVEDIR" \
      --species1_name   "$SPECIES1" \
      --species2_name   "$SPECIES2" \
      --beta_kl         "$BETA_KL" \
      $VAE_FLAG \
      --alpha           "$ALPHA" \
      --beta_cls        "$BETA_CLS" \
      --beta_mmd        "$BETA_MMD" \
      --beta_coral      "$BETA_CORAL" \
      --epochs          "$EPOCHS" \
      $CONDITION_FLAG \
      $DENOISE_FLAG \
      $IDENTITY_GRAPH_FLAG \
      --lr              "$LR" \
      $PREPROCESS_FLAG

    # clear PREPROCESS after first use if you only want it once
    PREPROCESS_FLAG=""
  done
done
