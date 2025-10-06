#!/usr/bin/env python3
"""Batch inference script for Boltz model on multiple YAML files."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from pipeline import collate, preprocess, transfer_batch_to_device


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = False
    alignment_reverse_diff: bool = False
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


def get_yaml_files(input_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Get all YAML files from a directory."""
    yaml_files = sorted(input_dir.glob("*.yaml"))
    if limit:
        yaml_files = yaml_files[:limit]
    return yaml_files


def preprocess_batch(
    yaml_files: List[Path],
    cache_dir: str | Path = "~/.boltz",
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    verbose: bool = False,
) -> List[dict]:
    """Preprocess multiple YAML files."""
    features_list = []
    
    iterator = tqdm(yaml_files, desc="Preprocessing") if verbose else yaml_files
    for yaml_file in iterator:
        try:
            features = preprocess(
                yaml_file,
                cache_dir=cache_dir,
                use_msa_server=use_msa_server,
                msa_server_url=msa_server_url,
            )
            features_list.append(features)
        except Exception as e:
            print(f"Error preprocessing {yaml_file}: {e}")
            continue
    
    return features_list


def save_results(
    outputs: dict,
    yaml_files: List[Path],
    output_dir: Path,
    batch_idx: int,
):
    """Save inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = len(yaml_files)
    
    # Save metadata
    metadata = {
        "batch_idx": batch_idx,
        "files": [str(f) for f in yaml_files],
        "batch_size": batch_size,
    }
    
    # Save outputs for each item in the batch
    for i in range(batch_size):
        if i >= len(yaml_files):
            break
            
        yaml_file = yaml_files[i]
        item_name = yaml_file.stem
        item_dir = output_dir / f"batch_{batch_idx:04d}" / item_name
        item_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each output tensor/value
        for key, value in outputs.items():
            if key == "record":
                continue
                
            try:
                if isinstance(value, torch.Tensor):
                    # Handle batch dimension
                    if value.ndim > 0 and value.shape[0] >= i + 1:
                        item_value = value[i].cpu()
                        torch.save(item_value, item_dir / f"{key}.pt")
                elif isinstance(value, list) and len(value) > i:
                    # Save list items
                    with open(item_dir / f"{key}.json", "w") as f:
                        json.dump(value[i], f, indent=2, default=str)
            except Exception as e:
                print(f"Error saving {key} for {item_name}: {e}")
                continue
        
        # Save metadata for this item
        with open(item_dir / "metadata.json", "w") as f:
            json.dump({
                "yaml_file": str(yaml_file),
                "batch_idx": batch_idx,
                "batch_position": i,
            }, f, indent=2)
    
    # Save batch metadata
    batch_meta_file = output_dir / f"batch_{batch_idx:04d}" / "batch_metadata.json"
    with open(batch_meta_file, "w") as f:
        json.dump(metadata, f, indent=2)


def run_batch_inference(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 4,
    cache_dir: str = "~/.boltz",
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    recycling_steps: int = 0,
    sampling_steps: int = 100,
    diffusion_samples: int = 1,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    compile_mode: Optional[str] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
):
    """Run batch inference on all YAML files in a directory."""
    
    # Get all YAML files
    yaml_files = get_yaml_files(input_dir, limit=max_files)
    if not yaml_files:
        print(f"No YAML files found in {input_dir}")
        return
    
    print(f"Found {len(yaml_files)} YAML files")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {(len(yaml_files) + batch_size - 1) // batch_size}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    from trunk_inference_model import Boltz2TrunkInfer
    
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=True,
    )
    
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    steering_args = BoltzSteeringParams()
    
    model = Boltz2TrunkInfer.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    
    # Optionally compile model
    if compile_mode and compile_mode.lower() != "none":
        print(f"Compiling model with mode: {compile_mode}")
        model = torch.compile(model, mode=compile_mode, dynamic=False)
    
    # Process in batches
    total_time = 0
    total_preprocess_time = 0
    total_inference_time = 0
    num_batches = (len(yaml_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(yaml_files))
        batch_files = yaml_files[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)")
        print(f"{'='*60}")
        
        # Preprocess
        preprocess_start = time.time()
        features_list = preprocess_batch(
            batch_files,
            cache_dir=cache_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            verbose=verbose,
        )
        preprocess_time = time.time() - preprocess_start
        total_preprocess_time += preprocess_time
        print(f"Preprocessing: {preprocess_time:.2f}s ({preprocess_time/len(batch_files):.2f}s per file)")
        
        if not features_list:
            print(f"No valid features for batch {batch_idx}, skipping...")
            continue
        
        # Collate into batch
        try:
            batch = collate(features_list)
            batch = transfer_batch_to_device(batch, device)
        except Exception as e:
            print(f"Error collating batch {batch_idx}: {e}")
            continue
        
        # Inference
        inference_start = time.time()
        with torch.inference_mode():
            outputs = model(batch, recycling_steps=recycling_steps)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        print(f"Inference: {inference_time:.2f}s ({inference_time/len(batch_files):.2f}s per file)")
        
        # Save results
        save_start = time.time()
        save_results(outputs, batch_files, output_dir, batch_idx)
        save_time = time.time() - save_start
        print(f"Saving: {save_time:.2f}s")
        
        batch_time = preprocess_time + inference_time + save_time
        total_time += batch_time
        print(f"Total batch time: {batch_time:.2f}s")
        
        # Clean up
        del batch, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(yaml_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total preprocessing: {total_preprocess_time:.2f}s ({total_preprocess_time/len(yaml_files):.2f}s per file)")
    print(f"Total inference: {total_inference_time:.2f}s ({total_inference_time/len(yaml_files):.2f}s per file)")
    print(f"Average time per file: {total_time/len(yaml_files):.2f}s")
    print(f"Output directory: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference with Boltz model on multiple YAML files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default="samples",
        help="Directory containing YAML files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="batch_outputs",
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("~/.boltz/boltz2_conf.ckpt").expanduser(),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples to process in each batch.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.boltz",
        help="Cache directory for Boltz data.",
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        default=True,
        help="Use MSA server for generating MSAs.",
    )
    parser.add_argument(
        "--no-msa-server",
        action="store_true",
        help="Disable MSA server.",
    )
    parser.add_argument(
        "--msa-server-url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server URL.",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=0,
        help="Number of recycling steps.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=100,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=1,
        help="Number of diffusion samples.",
    )
    parser.add_argument(
        "--subsample-msa",
        action="store_true",
        default=True,
        help="Subsample MSA sequences.",
    )
    parser.add_argument(
        "--num-subsampled-msa",
        type=int,
        default=1024,
        help="Number of MSA sequences to keep after subsampling.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune", "none"],
        help="Torch compile mode. Use 'none' to disable compilation.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with progress bars.",
    )
    
    args = parser.parse_args()
    
    # Handle MSA server flag
    use_msa_server = args.use_msa_server and not args.no_msa_server
    
    run_batch_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_msa_server=use_msa_server,
        msa_server_url=args.msa_server_url,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        subsample_msa=args.subsample_msa,
        num_subsampled_msa=args.num_subsampled_msa,
        compile_mode=args.compile_mode,
        max_files=args.max_files,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

