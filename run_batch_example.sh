#!/bin/bash
# Example batch inference runs

# Set common variables
CHECKPOINT="${HOME}/.boltz/boltz2_conf.ckpt"
INPUT_DIR="samples"

echo "=========================================="
echo "Boltz Batch Inference Examples"
echo "=========================================="
echo ""

# Example 1: Quick test with 4 files
echo "Example 1: Quick test (4 files, batch size 2)"
python batch_inference.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "test_output" \
    --checkpoint "${CHECKPOINT}" \
    --batch-size 2 \
    --max-files 4 \
    --verbose

echo ""
echo "=========================================="
echo ""

# Example 2: Small batch run without MSA server
echo "Example 2: Batch inference without MSA server (8 files)"
python batch_inference.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "batch_no_msa" \
    --checkpoint "${CHECKPOINT}" \
    --batch-size 4 \
    --max-files 8 \
    --no-msa-server \
    --verbose

echo ""
echo "=========================================="
echo ""

# Example 3: Optimized batch with torch compile
echo "Example 3: Optimized inference with torch.compile (8 files)"
python batch_inference.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "batch_compiled" \
    --checkpoint "${CHECKPOINT}" \
    --batch-size 4 \
    --max-files 8 \
    --compile-mode default \
    --verbose

echo ""
echo "=========================================="
echo ""

# Example 4: Production-like run (uncomment to use)
# echo "Example 4: Full production run (all files)"
# python batch_inference.py \
#     --input-dir "${INPUT_DIR}" \
#     --output-dir "production_output_$(date +%Y%m%d_%H%M%S)" \
#     --checkpoint "${CHECKPOINT}" \
#     --batch-size 8 \
#     --compile-mode default \
#     --sampling-steps 200 \
#     --verbose

echo ""
echo "All examples completed!"

