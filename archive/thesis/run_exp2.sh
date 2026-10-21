#!/bin/bash
set -e

PROJ=/home/liris/sounania/projects/hatexplain_bert
source $PROJ/venv/bin/activate
cd $PROJ/src
mkdir -p ../outputs/logs

for SEED in 0 1 2; do
  echo "=============================================="
  echo " SEED $SEED — Experiment 2 (all-layer FT)"
  echo "=============================================="
  python model_finetuning_alllayer.py --seed $SEED \
      2>&1 | tee ../outputs/logs/ft-alllayer-s$SEED.log

  python hate_prototypes_bert.py --seed $SEED --tag alllayer \
      --checkpoint_dir ../outputs/checkpoints/hatexplain-alllayer-bert-cased-s$SEED \
      2>&1 | tee ../outputs/logs/eval-alllayer-s$SEED.log
done

echo "Experiment 2 complete."
ls -1 ../outputs/predictions/alllayer-*
