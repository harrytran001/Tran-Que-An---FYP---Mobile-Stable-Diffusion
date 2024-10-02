# Import necessary modules
import torch
import random
import numpy as np

from helpers import CustomStableDiffusionPipeline


# Function to set seed for all random operations, ensuring reproducibility
def seed_everywhere(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Define model_id for the pretrained model from Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"

# Instantiate pipeline with the pretrained model
pipe = CustomStableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32
).to("cpu")

# Define parameters for image generation
seed = 42
seed_everywhere(seed)
step = 15
scale = 7.5
savepath = "seed" + str(seed) + "step" + str(step)

prompt = "apple on the table, realistic, highly detailed, vibrant colors, natural lighting, shiny surface, wooden table, shadows, high quality, photorealistic, masterpiece"
negative_prompt = "blurry, deformed, bad anatomy, disfigured, poorly drawn, mutation, mutated, extra limb, ugly, missing limb, blurry, floating, disconnected, malformed, blur, out of focus, bad lighting, unrealistic, text"
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=step,
    guidance_scale=scale,
).images[0]

image.save("results/" + savepath + prompt + "_final.png")  # Save the generated image
