import torch
import numpy as np
from ray import serve
import time
from dataclasses import asdict

from trunk_inference_model import Boltz2TrunkInfer
from pipeline import collate, transfer_batch_to_device, Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1, "num_cpus": 1})
class Boltz2TrunkInferModel:
    def __init__(self, checkpoint_path: str):
        subsample_msa = True
        num_subsampled_msa = 1024
        use_potentials = False
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5
        pairformer_args = PairformerArgsV2()

        msa_args = MSAModuleArgs(
            subsample_msa=subsample_msa,
            num_subsampled_msa=num_subsampled_msa,
            use_paired_feature=True,
        )

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials
        print(f"Loading model from {checkpoint_path}")
        self.model = Boltz2TrunkInfer.load_from_checkpoint(checkpoint_path, 
            strict=True,
            map_location="cuda",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )

    @torch.inference_mode()
    @serve.batch(max_batch_size=2, batch_wait_timeout_s=0.25)
    def forward_trunk(self, batch_feats: dict[str, torch.Tensor], recycling_steps: int = 0):
        start = time.time()
        batch_feats = collate(batch_feats)
        for k, v in batch_feats.items():
            batch_feats[k] = v.to("cuda")
        batch_feats = transfer_batch_to_device(batch_feats, "cuda")
        print(f"Collate and transfer to device finished in {time.time() - start:.2f}s")
        start = time.time()
        s, z, pdistogram = self.model(batch_feats, recycling_steps=recycling_steps)
        print(f"Inference finished in {time.time() - start:.2f}s")
        return s.cpu(), z.cpu(), pdistogram.cpu()