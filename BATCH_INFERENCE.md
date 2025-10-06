# Batch Inference for Boltz

This guide explains how to run batch inference on multiple YAML files using the `batch_inference.py` script.

## Overview

The batch inference script allows you to process multiple YAML input files in batches, which is more efficient than processing them one at a time. It automatically:
- Loads and preprocesses multiple YAML files
- Batches them together for parallel GPU processing
- Runs inference on each batch
- Saves results organized by batch and file

## Basic Usage

### Simple Example

Process all YAML files in the `samples` directory with a batch size of 4:

```bash
python batch_inference.py \
    --input-dir samples \
    --output-dir batch_outputs \
    --batch-size 4 \
    --checkpoint ~/.boltz/boltz2_conf.ckpt
```

### Process Limited Number of Files

For testing, process only the first 8 files:

```bash
python batch_inference.py \
    --input-dir samples \
    --output-dir batch_outputs \
    --batch-size 4 \
    --max-files 8 \
    --verbose
```

### With Torch Compile

Use torch.compile for faster inference (after warmup):

```bash
python batch_inference.py \
    --input-dir samples \
    --output-dir batch_outputs \
    --batch-size 4 \
    --compile-mode default
```

## Command Line Arguments

### Required Arguments

- `--input-dir`: Directory containing YAML files to process (default: `samples`)
- `--output-dir`: Directory to save results (default: `batch_outputs`)
- `--checkpoint`: Path to model checkpoint (default: `~/.boltz/boltz2_conf.ckpt`)

### Batch Processing

- `--batch-size`: Number of samples per batch (default: 4)
  - Larger batches = faster processing but more GPU memory
  - Adjust based on your GPU memory
- `--max-files`: Maximum number of files to process (useful for testing)

### Model Parameters

- `--recycling-steps`: Number of recycling iterations (default: 0)
- `--sampling-steps`: Diffusion sampling steps (default: 100)
- `--diffusion-samples`: Number of samples to generate (default: 1)

### MSA Options

- `--use-msa-server`: Enable MSA server (default: True)
- `--no-msa-server`: Disable MSA server
- `--msa-server-url`: MSA server URL (default: https://api.colabfold.com)
- `--subsample-msa`: Subsample MSA sequences (default: True)
- `--num-subsampled-msa`: Number of MSA sequences to keep (default: 1024)

### Performance

- `--compile-mode`: Torch compile mode for optimization
  - `default`: Balanced speed/compilation time
  - `reduce-overhead`: Faster for small models
  - `max-autotune`: Maximum optimization (slow compilation)
  - `none`: Disable compilation

### Other

- `--cache-dir`: Cache directory for Boltz data (default: `~/.boltz`)
- `--verbose`: Enable progress bars and detailed output

## Output Structure

Results are saved in the following structure:

```
batch_outputs/
├── batch_0000/
│   ├── batch_metadata.json
│   ├── P23975_ligand_0029/
│   │   ├── metadata.json
│   │   ├── pdistogram.pt
│   │   ├── plddt.pt
│   │   └── ... (other outputs)
│   └── P23975_ligand_0031/
│       └── ...
├── batch_0001/
│   └── ...
└── ...
```

Each file's results include:
- `metadata.json`: Information about the input file and batch
- `*.pt`: PyTorch tensors with model outputs (pdistogram, plddt, etc.)

## Performance Tips

1. **Batch Size**: 
   - Start with batch size 2-4 and increase if you have GPU memory
   - Monitor GPU memory usage with `nvidia-smi`

2. **Compilation**:
   - First run will be slow (compilation overhead)
   - Subsequent batches will be much faster
   - Use `--compile-mode default` for best results

3. **MSA Processing**:
   - If MSAs are pre-computed, they'll load faster
   - MSA server queries are the slowest part of preprocessing

4. **Testing**:
   - Use `--max-files 4` to test on a small subset first
   - Check output structure and timing before full run

## Example Workflows

### Quick Test Run

```bash
# Test with 4 files, batch size 2
python batch_inference.py \
    --max-files 4 \
    --batch-size 2 \
    --verbose
```

### Production Run

```bash
# Process all files with optimal settings
python batch_inference.py \
    --input-dir samples \
    --output-dir results_$(date +%Y%m%d_%H%M%S) \
    --batch-size 8 \
    --compile-mode default \
    --sampling-steps 200 \
    --verbose
```

### High-Quality Predictions

```bash
# Maximum quality with recycling
python batch_inference.py \
    --input-dir samples \
    --output-dir high_quality_results \
    --batch-size 4 \
    --recycling-steps 3 \
    --sampling-steps 200 \
    --diffusion-samples 5
```

## Monitoring Progress

The script provides detailed timing information:
- Preprocessing time per file
- Inference time per batch
- Average time per file
- Total time and throughput

Example output:
```
============================================================
Batch 1/10 (4 files)
============================================================
Preprocessing: 45.32s (11.33s per file)
Inference: 23.45s (5.86s per file)
Saving: 2.15s
Total batch time: 70.92s

============================================================
SUMMARY
============================================================
Total files processed: 40
Total time: 650.23s
Total preprocessing: 450.12s (11.25s per file)
Total inference: 180.45s (4.51s per file)
Average time per file: 16.26s
```

## Troubleshooting

### Out of Memory

If you get CUDA out of memory errors:
1. Reduce `--batch-size`
2. Reduce `--num-subsampled-msa`
3. Reduce `--sampling-steps`

### Slow Preprocessing

If preprocessing is slow:
1. Check MSA server connectivity
2. Pre-compute MSAs and place them with YAML files
3. Use `--no-msa-server` if MSAs are provided

### Compilation Issues

If torch.compile fails:
1. Use `--compile-mode none` to disable
2. Check PyTorch version (2.0+ required for compile)
3. Update CUDA drivers if needed

