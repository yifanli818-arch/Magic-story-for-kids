import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import torch
import random
import os

# --- 页面 UI 设置 ---
st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")
st.write("Upload a photo and I will tell you a wonderful fairy tale (70-100 words)!")

# --- 1. 加载视觉模型 ---
@st.cache_resource
def load_vision_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

processor, model, device = load_vision_model()

# --- 2. 增强版多分支长故事引擎 ---
def generate_perfect_story(caption):
    c = caption.lower()

    # 分支 1: 运动场景 (Sports / Running / Playing)
    if any(word in c for word in ["run", "play", "ball", "sport", "jump", "kick", "football", "basketball"]):
        templates = [
            f"Under the bright golden sun, the exciting scene of {caption} filled the park with pure energy. A brave young child wearing magic sneakers joined the game and discovered that every time they jumped, they could almost touch the fluffy white clouds. The wind cheered like a happy crowd, and the grass felt like a soft green carpet beneath their fast-moving feet. The child learned that staying active and playing with friends is the most powerful secret to long-lasting happiness and strength. It was a day of victory and laughter. The end."
        ]
    # 分支 2: 建筑/城堡 (Castle / Building)
    elif any(word in c for word in ["castle", "building", "house", "tower", "city"]):
        templates = [
            f"High above the misty emerald hills, the {caption} stood proudly like a giant's golden crown. One sunny morning, a brave little child found a glowing silver key hidden deep in the tall grass and slowly opened a heavy secret door. Inside, they discovered a magical library filled with thousands of flying books that whispered wonderful stories about the bright stars and moon. The child stayed there all afternoon, learning that every ancient stone building holds a beautiful secret adventure for those who believe in magic. The end."
        ]
    # 分支 3: 动物相关 (Animals)
    elif any(animal in c for animal in ["dog", "cat", "bird", "animal", "bear", "lion", "elephant"]):
        templates = [
            f"In a peaceful garden filled with whispering colorful flowers, the {caption} was searching for a new best friend to play with. A kind little child brought a small wooden basket filled with magic sweet berries to share under the shade of a giant oak tree. They spent the entire golden afternoon dancing together under a bright, glowing rainbow. The child learned that being gentle and kind to every animal is the greatest magic in the whole world. They promised to be best friends forever and ever. The end."
        ]
    # 分支 4: 食物相关 (Food / Eating)
    elif any(word in c for word in ["food", "eat", "cake", "fruit", "dinner", "table", "cooking"]):
        templates = [
            f"On a big wooden table decorated with sparkling lights, the delicious {caption} looked like a feast from a fairy tale kingdom. A hungry child took a tiny bite and suddenly realized that this special food gave them the power to understand the language of birds. Each flavor was a different musical note, turning the meal into a beautiful symphony of joy. The family laughed together, realizing that sharing a good meal is how we share our love and create the warmest memories. The end."
        ]
    # 分支 5: 人物/小孩 (Person / Child / People)
    elif any(word in c for word in ["person", "child", "girl", "boy", "man", "woman"]):
        templates = [
            f"Once upon a time, the lovely {caption} was exploring the edge of a magical crystal forest. With a heart full of curiosity, the child followed a trail of shimmering blue butterflies that led to a hidden pond of wishes. By simply smiling at the water, the child made the trees bloom with silver leaves and golden fruits. It was a day to remember that a kind soul and a big smile can change the whole world into a better and more beautiful place for everyone. The end."
        ]
    # 分支 6: 通用/自然场景 (General / Nature)
    else:
        templates = [
            f"Once upon a time in a faraway land, there was {caption} that sparkled with a mysterious and friendly light. The morning sky was painted with soft pink clouds and the green grass tasted exactly like delicious sweet mint candy. A happy child arrived with a silver whistle and taught the little blue birds how to sing a beautiful new tune. It was a wonderful and peaceful day where everyone lived together in perfect joy. It was truly a magical place where every dream could come true. The end."
        ]

    return random.choice(templates)

# --- 3. 网页交互逻辑 ---
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, caption="Your Uploaded Image", use_container_width=True)

    if st.button("✨ Generate Magic Story & Audio"):
        with st.spinner("Writing your long fairy tale..."):
            # A. 识别图片
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # B. 生成长故事
            story_text = generate_perfect_story(caption)
            
            # C. 统计单词数
            word_count = len(story_text.split())

            # D. 生成语音
            tts = gTTS(text=story_text, lang='en')
            audio_file = "story.mp3"
            tts.save(audio_file)

            # E. 展示结果
            st.subheader(f"📖 Story ({word_count} words)")
            st.success(story_text)

            st.subheader("🎙️ Audio")
            st.audio(audio_file)

            # 撒花庆祝
            st.balloons()
