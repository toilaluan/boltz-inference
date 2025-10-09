from pydantic import BaseModel
from fastapi import FastAPI
from ray import serve
from pipeline import preprocess
from ray_model import Boltz2TrunkInferModel

class InferenceRequest(BaseModel):
    yaml_file: str
    recycling_steps: int = 0

app = FastAPI()


@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}, num_replicas=4, max_ongoing_requests=1024
)
@serve.ingress(app)
class InferenceService:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    @app.post("/inference")
    def inference(self, request: InferenceRequest):
        inputs = preprocess(request.yaml_file)
        outputs = self.model_handler.forward_trunk(inputs, recycling_steps=request.recycling_steps)
        return outputs

print("Binding model")
inferencer = InferenceService.bind(Boltz2TrunkInferModel.bind("~/.boltz/boltz2_conf.ckpt"))
print("Model bound")

if __name__ == "__main__":
    import ray
    import os

    if not ray.is_initialized():
        print("Initializing Ray")
        ray.init(
            address=os.getenv("RAY_ADDRESS", "auto"),
            namespace="boltz2_data",
        )
    print("Starting Ray Serve")
    serve.run(
        inferencer,
        name="boltz2_data_service",
        route_prefix="/",
    )