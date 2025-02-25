from diffusers import AutoPipelineForText2Image,DiffusionPipeline
from utils import device
import torch


pipeline = AutoPipelineForText2Image.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True).to(device)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt).images[0]