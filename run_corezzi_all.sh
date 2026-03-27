#!/bin/zsh
# Run all 12 NMR Corezzi MCMC fits sequentially.
# Usage: bash run_corezzi_all.sh

cd "$(dirname "$0")"

SAMPLES=(EDA  EDA  EDA  DAP  DAP  DAP  DAP2 DAP2 DAP2 DAB  DAB  DAB)
TEMPS=(  25C  33C  40C  25C  33C  40C  25C  33C  40C  25C  33C  40C)

TOTAL=${#SAMPLES[@]}

for (( i=0; i<TOTAL; i++ )); do
    SAMPLE=${SAMPLES[$i]}
    TEMP=${TEMPS[$i]}
    COUNT=$((i + 1))
    echo "=== [$COUNT/$TOTAL] $SAMPLE $TEMP — $(date) ==="
    python BatchBayesian_nmr_corezzi.py --mcmc "$SAMPLE" "$TEMP"
    echo "=== Done: $SAMPLE $TEMP — $(date) ==="
done

echo "All done at $(date)"
