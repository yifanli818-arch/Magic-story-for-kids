%%writefile app.py
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import torch
import random
import os

# --- 页面配置 ---
st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")
st.write("Upload a photo and listen to a wonderful story with over 50 words!")

# --- 1. 加载视觉模型 ---
@st.cache_resource
def load_vision_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return proc, model, device

processor, model, device = load_vision_model()

# --- 2. 增强版长故事引擎 (确保 > 50 字) ---
def generate_perfect_story(caption):
    c = caption.lower()
    
    # 场景 A: 建筑/城堡
    if any(word in c for word in ["castle", "building", "house", "tower"]):
        templates = [
            f"High above the emerald hills, the {caption} stood like a giant's golden crown under the bright sun. A lucky child found a shining silver key hidden in the tall grass and slowly opened a secret door. Inside, they discovered a magical room filled with flying books that told wonderful stories about the stars. The child stayed there all day, learning that every ancient building holds a beautiful secret adventure for those who dare to look. The end."
        ]
    # 场景 B: 动物
    elif any(word in c for word in ["dog", "cat", "bird", "animal", "bear", "horse"]):
        templates = [
            f"In a beautiful garden of whispering colorful flowers, the {caption} was looking for a new best friend to play with. A kind little child brought a basket of magic sweet berries to share under the big oak tree. They spent the entire afternoon dancing happily under a bright rainbow. The child learned that being gentle and kind to all animals is the greatest magic in the whole world. Everyone was very happy. The end."
        ]
    # 场景 C: 其他/通用
    else:
        templates = [
            f"Once upon a time in a land far away, there was {caption} that glowed with a mysterious light. The sky was painted with soft pink clouds and the green grass tasted exactly like sweet mint candy. A happy child arrived and taught the little birds how to whistle a beautiful new tune. It was a wonderful day where everyone lived together in peace and joy forever. It was truly a magical place to be. The end."
        ]
    
    return random.choice(templates)

# --- 3. 网页交互逻辑 ---
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, use_container_width=True)
    
    if st.button("✨ Generate Long Story & Audio"):
        with st.spinner("Writing a long magical story for you..."):
            # A. 识别图片
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # B. 生成长故事
            story_text = generate_perfect_story(caption)
            
            # C. 统计字数（展示给用户看，确保达标）
            word_count = len(story_text.split())
            
            # D. 生成语音
            tts = gTTS(text=story_text, lang='en')
            audio_file = "long_story.mp3"
            tts.save(audio_file)
            
            # E. 展示结果
            st.subheader(f"📖 Story ({word_count} words)")
            st.success(story_text)
            
            st.subheader("🎙️ Audio")
            st.audio(audio_file)
            st.balloons()
