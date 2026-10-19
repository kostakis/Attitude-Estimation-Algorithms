#!/bin/bash
# Run all 4 LSTM evaluations and generate plots
# Usage: ./run_all_evals.sh [sample_index]

SAMPLE=${1:-50}
RESULTS_DIR="Results"
LSTM_DIR="Results/LSTM"

mkdir -p "$LSTM_DIR"

echo "======================================"
echo "Running all 4 LSTM evaluations"
echo "Sample index for plots: $SAMPLE"
echo "======================================"

# Define experiments: name | binary | model | window_size
experiments=(
    "attn_N10|./lstmEval.out|lstm_model_epoch_500_datasplitatten.pt|10"
    "attn_N30|./lstmEval.out|lstm_model_epoch_500_datasplitatten30pred.pt|30"
    "noattn_N10|./lstmEvalNoAttn.out|lstm_model_epoch_500_datasplitNOAtten.pt|10"
    "noattn_N30|./lstmEvalNoAttn.out|lstm_model_epoch_500_datasplitNOAtten30pred.pt|30"
)

for exp in "${experiments[@]}"; do
    IFS='|' read -r name binary model window <<< "$exp"

    echo ""
    echo "======================================"
    echo "Experiment: $name (N=$window)"
    echo "Model: $model"
    echo "Binary: $binary"
    echo "======================================"

    # Run evaluation and save predictions
    $binary "$model" --save-all -w "$window"

    # Rename predictions CSV
    mv "$RESULTS_DIR/lstm_predictions.csv" "$RESULTS_DIR/lstm_predictions_${name}.csv"
    echo "Saved: $RESULTS_DIR/lstm_predictions_${name}.csv"

    # Generate plot (script reads from Results/lstm_predictions.csv)
    cp "$RESULTS_DIR/lstm_predictions_${name}.csv" "$RESULTS_DIR/lstm_predictions.csv"
    python3 plot_single_pred.py "$SAMPLE"

    # Rename plot to include experiment name
    generated_plot="$LSTM_DIR/single_sample_prediction_N${window}_test_${SAMPLE}.png"
    final_plot="$LSTM_DIR/single_sample_${name}_test_${SAMPLE}.png"
    if [ -f "$generated_plot" ]; then
        mv "$generated_plot" "$final_plot"
        echo "Saved plot: $final_plot"
    fi
done

# Clean up temp file
rm -f "$RESULTS_DIR/lstm_predictions.csv"

echo ""
echo "======================================"
echo "All done! Generated files:"
echo "======================================"
ls -la "$RESULTS_DIR"/lstm_predictions_*.csv
ls -la "$LSTM_DIR"/single_sample_*_test_${SAMPLE}.png
