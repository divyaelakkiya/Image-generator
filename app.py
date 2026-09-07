import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("AI Image Generator")

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5"
    )

    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt).images[0]

    st.image(image, caption="Generated Image")