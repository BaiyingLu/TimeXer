#!/usr/bin/env bash
# Run all three per-dataset test scripts sequentially and rename result
# directories so outputs don't overwrite each other.
#
# Usage:  bash scripts/forecast_exogenous/combined/test_all.sh

set -e  # stop on first error

SETTING="long_term_forecast_combined_24_6_TimeXer_combined_ftMS_sl24_ll6_pl6_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_TimeXer-BGlucose-combined"
RESULTS="./results"

for dataset in ohio bris hupa; do
    echo ""
    echo "========================================"
    echo "Testing: ${dataset}"
    echo "========================================"

    bash "$(dirname "$0")/test_${dataset}.sh"

    # Move results to dataset-specific directory (overwrite if already exists)
    rm -rf "${RESULTS}/${SETTING}_${dataset}"
    mv "${RESULTS}/${SETTING}" "${RESULTS}/${SETTING}_${dataset}"

    echo "Saved to: results/${SETTING}_${dataset}/"
done

echo ""
echo "All done. Results in:"
ls -d ${RESULTS}/${SETTING}_*
