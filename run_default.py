from boltz.main import predict
import time


input_dir = "test_samples"
output_dir = "outputs"
affinity_checkpoint = "/root/.boltz/boltz2_aff.ckpt"
batch_size = 1
max_files = 1
verbose = True

recycling_steps=3
sampling_steps=100
sampling_steps_affinity=100
diffusion_samples=1
diffusion_samples_affinity=3

start_time = time.time()
predict(
    data=input_dir,
    out_dir=output_dir,
    affinity_checkpoint=affinity_checkpoint,
    devices=1,
    accelerator="gpu",
    recycling_steps=recycling_steps,
    sampling_steps=sampling_steps,
    diffusion_samples=diffusion_samples,
    sampling_steps_affinity=sampling_steps_affinity,
    diffusion_samples_affinity=diffusion_samples_affinity,
    affinity_mw_correction=True,
    override=True,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


