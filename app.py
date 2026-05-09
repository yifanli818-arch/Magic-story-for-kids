import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import torch
import random
import os

# --- Page Configuration ---
st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")
st.write("Upload a photo and I will tell you a wonderful fairy tale!")

# --- 1. Load Vision Model ---
@st.cache_resource
def load_vision_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

processor, model, device = load_vision_model()

# --- 2. Story Engine (70-100 Words, Starting with Once Upon a Time) ---
def generate_perfect_story(caption):
    c = caption.lower()

    # Branch 1: Sports & Exercise
    if any(word in c for word in ["run", "play", "ball", "sport", "jump", "kick", "football", "basketball", "exercise"]):
        templates = [
            f"Once upon a time, under a bright golden sun, the lovely {caption} filled the world with pure energy. A brave young child wearing magic sneakers joined the fun and discovered that every time they jumped, they could almost touch the fluffy white clouds. The wind cheered like a happy friend, and the soft grass felt like a green carpet beneath their fast feet. The child learned that staying active and playing outside is the most powerful secret to long-lasting happiness and great strength. It was a day of victory and laughter for everyone. The end."
        ]
    # Branch 2: Castles & Buildings
    elif any(word in c for word in ["castle", "building", "house", "tower", "city"]):
        templates = [
            f"Once upon a time, high above the misty emerald hills, the {caption} stood proudly like a giant's golden crown. One sunny morning, a brave little child found a glowing silver key hidden deep in the tall grass and slowly opened a heavy secret door. Inside, they discovered a magical library filled with thousands of flying books that whispered wonderful stories about the bright stars and the silver moon. The child stayed there all day, learning that every ancient stone building holds a beautiful secret adventure for those who truly believe in magic. The end."
        ]
    # Branch 3: Animals
    elif any(animal in c for animal in ["dog", "cat", "bird", "animal", "bear", "lion", "elephant", "rabbit"]):
        templates = [
            f"Once upon a time, in a peaceful garden filled with whispering colorful flowers, the {caption} was searching for a new best friend to play with. A kind little child brought a small wooden basket filled with magic sweet berries to share under the shade of a giant oak tree. They spent the entire golden afternoon dancing together under a bright, glowing rainbow in the sky. The child learned that being gentle and kind to every animal is the greatest magic in the whole world. They promised to be best friends forever. The end."
        ]
    # Branch 4: Food
    elif any(word in c for word in ["food", "eat", "cake", "fruit", "dinner", "table", "cooking", "bread"]):
        templates = [
            f"Once upon a time, on a big wooden table decorated with sparkling lights, the delicious {caption} looked like a feast from a fairy tale kingdom. A hungry child took a tiny bite and suddenly realized that this special food gave them the magic power to understand the language of birds. Each flavor was a different musical note, turning the meal into a beautiful symphony of joy. The family laughed together, realizing that sharing a good meal is how we share our love and create the warmest memories in our hearts. The end."
        ]
    # Branch 5: People
    elif any(word in c for word in ["person", "child", "girl", "boy", "man", "woman"]):
        templates = [
            f"Once upon a time, the lovely {caption} was exploring the edge of a magical crystal forest. With a heart full of curiosity, the child followed a trail of shimmering blue butterflies that led to a hidden pond of magic wishes. By simply smiling at the water, the child made the trees bloom with silver leaves and golden fruits. It was a day to remember that a kind soul and a big happy smile can change the whole world into a better and more beautiful place for everyone to live. The end."
        ]
    # Branch 6: General Nature
    else:
        templates = [
            f"Once upon a time, in a faraway land, there was {caption} that sparkled with a mysterious and friendly light. The morning sky was painted with soft pink clouds and the green grass tasted exactly like delicious sweet mint candy. A happy child arrived with a silver whistle and taught the little blue birds how to sing a beautiful new tune. It was a wonderful and peaceful day where everyone lived together in perfect joy. It was truly a magical place where every dream could come true if you just believed. The end."
        ]

    return random.choice(templates)

# --- 3. UI Logic ---
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, caption="Your Uploaded Image", use_container_width=True)

    if st.button("✨ Generate Magic Story"):
        with st.spinner("Writing your long fairy tale..."):
            # A. Image Recognition
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # B. Story Generation
            story_text = generate_perfect_story(caption)
            
            # C. Word Count
            word_count = len(story_text.split())

            # D. Audio Generation
            tts = gTTS(text=story_text, lang='en')
            audio_file = "story.mp3"
            tts.save(audio_file)

            # E. Results
            st.subheader(f"📖 Story ({word_count} words)")
            st.success(story_text)

            st.subheader("🎙️ Listen to the Magic")
            st.audio(audio_file)

            st.balloons()
