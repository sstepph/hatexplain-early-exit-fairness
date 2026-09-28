#!/bin/bash
set -e

PROJ=/home/liris/sounania/projects/hatexplain_bert
source $PROJ/venv/bin/activate
cd $PROJ/src
mkdir -p ../outputs/logs

for SEED in 0 1 2; do
  echo "=============================================="
  echo " SEED $SEED — Experiment 1 (standard FT)"
  echo "=============================================="
  python model_finetuning.py --seed $SEED \
      2>&1 | tee ../outputs/logs/ft-std-s$SEED.log

  python hate_prototypes_bert.py --seed $SEED --tag std \
      2>&1 | tee ../outputs/logs/eval-std-s$SEED.log
done

echo "Experiment 1 complete."
ls -1 ../outputs/predictions/std-*
