import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import torch
import random
import os

st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")

@st.cache_resource
def load_vision_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return proc, model, device

processor, model, device = load_vision_model()

def generate_perfect_story(caption):
    c = caption.lower()
    if "castle" in c or "building" in c or "house" in c:
        templates = [f"High above the emerald hills, the {caption} stood like a giant's crown. A lucky child found a silver key in the grass and opened a secret door. Inside, they found a room filled with flying books that told wonderful stories. The end."]
    elif any(animal in c for animal in ["dog", "cat", "bird", "animal"]):
        templates = [f"In a garden of whispering flowers, the {caption} was looking for a friend. A little child brought a basket of magic berries to share. They spent the afternoon dancing under a rainbow. The end."]
    else:
        templates = [f"Once upon a time, there was {caption}. The sky was painted with pink clouds and the grass tasted like sweet mint. Everyone lived happily ever after. The end."]
    return random.choice(templates)

uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, use_container_width=True)
    if st.button("✨ Generate Story & Audio"):
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        story_text = generate_perfect_story(caption)
        tts = gTTS(text=story_text, lang='en')
        tts.save("story.mp3")
        st.subheader("📖 Story")
        st.success(story_text)
        st.audio("story.mp3")
        st.balloons()
