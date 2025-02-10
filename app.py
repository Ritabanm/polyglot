import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from googletrans import Translator

# Load Hugging Face model and tokenizer
model_name = "bigscience/bloom-560m"  # Smaller version of BLOOM for free usage
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize translator
translator = Translator()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def translate_text(text, src_lang, dest_lang):
    """Translate text between languages."""
    try:
        return translator.translate(text, src=src_lang, dest=dest_lang).text
    except Exception as e:
        return f"Translation error: {e}"

def chat_with_llm(prompt, language):
    """Get response from LLM and translate it back."""
    try:
        translated_prompt = translate_text(prompt, language, "en")
        inputs = tokenizer(translated_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=512)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_response = translate_text(response_text, "en", language)
        return translated_response
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Multilingual Chatbot with Translation")
st.write("Chat with a multilingual model and translate responses instantly!")

# Select language
languages = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Bengali": "bn", "Chinese": "zh-cn"}
language = st.selectbox("Choose your language", list(languages.keys()))
language_code = languages[language]

# Display chat history
st.write("### Conversation")
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")

# User input
user_input = st.text_input("Your message")
if st.button("Send"):
    if user_input.strip():
        bot_response = chat_with_llm(user_input, language_code)
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
        st.experimental_rerun()  # Refresh the conversation
    else:
        st.warning("Please enter a message.")
