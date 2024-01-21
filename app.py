import option
import streamlit as st
import replicate
import os
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
import time


# Initialize the flag to track whether audio has been played
audio_played = False

# Global variables for model and tokenizer
gpt2_model = None
gpt2_tokenizer = None

# Home function
def home():
    col1, col2 = st.columns([2, 1])  # Adjust the ratio based on your preference
    with col1:
        st.write("## Welcome to the Amazing Chatbot Experience! 🌟")
        st.write("Unleash the power of conversation with our intelligent chatbot.")
        #st.image("your_image_url_here", caption="Chatbot Image", use_column_width=True)  # Add an image for visual appeal
        st.markdown("<style>div.Widget.row-widget.stButton > button{margin-left:auto;margin-right:0}</style>", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.9
    if "max_length" not in st.session_state:
        st.session_state.max_length = 500
    if "audio_played" not in st.session_state:
        st.session_state.audio_played = False  

# Function to generate LLaMA2 response
@st.cache_data(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    global gpt2_model, gpt2_tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to generate LLaMA2 response
@st.cache_data(show_spinner=False)
def generate_llama2_response_cached(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages):
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # Call the actual function without await
    output = replicate.run(llm, {
        "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "repetition_penalty": 1
    })

    return list(output)

# Function to generate LLaMA2 response
def generate_llama2_response(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages):
    return generate_llama2_response_cached(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages)

# Function to capture voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak something...")
        audio = recognizer.listen(source)

    try:
        text_input = recognizer.recognize_google(audio)
        st.success("Voice input recognized: " + text_input)
        return text_input
    except sr.UnknownValueError:
        st.warning("Sorry, I didn't catch that.")
        return None

# Function to play text as speech
def play_text_as_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_filename = f"response_{time.time()}.mp3"
    tts.save(audio_filename)
    st.audio(audio_filename, format="audio/mp3", start_time=0)

# Function to download audio
def download_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_filename = f"response_{time.time()}.mp3"
    tts.save(audio_filename)
    st.audio(audio_filename, format="audio/mp3", start_time=0)

def main():
    # Set Replicate API token
    replicate_api = "r8_0QpEN4c6M8jTFIT7cWWru7FpbpfeKvx0PaWaC"
    os.environ["REPLICATE_API_TOKEN"] = replicate_api

    # Set page configuration
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

    # Initialize session state
    init_session_state()

    # Load GPT-2 model and tokenizer
    if gpt2_model is None or gpt2_tokenizer is None:
        load_gpt2_model_and_tokenizer()

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Chatbot Menu')
        st.write('Choose a chatbot to interact with.')

        option = st.selectbox("Select a Chatbot", ["Home", "Llama 2", "User Manual", "About Us"])

    # User Manual page content
    if option == "User Manual":
        st.markdown("## 📚 User Manual")
        st.write("Welcome to the User Manual! Here's a step-by-step guide to using the chatbot:")

        # Step 1
        st.subheader("Step 1: Choose Your Chatbot")
        st.write("👉 Select the 'Llama 2' chatbot from the sidebar.")

        # Step 2
        st.subheader("Step 2: Ask Questions")
        st.write("🗣️ You can interact with the chatbot by typing questions or using voice input. Click the 'Ask Question in Voice' button to speak your question.")

        # Step 3
        st.subheader("Step 3: Receive Responses")
        st.write("💬 The chatbot will provide intelligent and contextually relevant responses to your questions. Responses will be displayed in the chat window.")

        # Step 4
        st.subheader("Step 4: Download Audio")
        st.write("🔊 If you want to save the chatbot's response as an audio file, click the 'Download Audio' button.")

        # Step 5
        st.subheader("Step 5: Clear Chat History")
        st.write("🧹 To start a new conversation, click the 'Clear Chat History' button. This will remove all previous messages.")

        # Step 6
        st.subheader("Step 6: Refresh Page (Optional)")
        st.write("🔄 If the chatbot is slow or unresponsive, you can refresh the page to start a fresh session.")

    if option == "Home":
        col1, col2 = st.columns([2, 1])  # Adjust the ratio based on your preference
        with col1:
            st.write("## Welcome to the Chatbot")
            st.write("Feel free to interact with the chatbot.")
            st.write("## Start your Amazing Chatbot Experience! 🌟")

            # Display the image
            st.image("https://images.prismic.io/intuzwebsite/d9daef05-a416-4e84-b0f8-2d5e2e3b58d8_A+Comprehensive+Guide+to+Building+an+AI+Chatbot%402x.png?w=2400&q=80&auto=format,compress&fm=png8", caption="Welcome")

    elif option == "Llama 2":
        st.subheader('Chatbot')
        # Only include Llama2-13B in the dropdown
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-13B'], key='selected_model')
        st.session_state.llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        st.session_state.temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        st.session_state.top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        st.session_state.max_length = st.sidebar.slider('max_length', min_value=32, max_value=1024, value=1024, step=8)
        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
        # st.write("This chatbot is powered by Hugging Face's transformers library. It uses the Llama 2-13B language model, which is designed to provide intelligent and contextually relevant responses in natural language conversations.")
        # st.write("Feel free to interact with the chatbot on the 'Home' page!")

    # About Us page content
    if option == "About Us":
        st.markdown("## About Us")
        # Show specific content when Llama2-13B is selected
        if st.session_state.llm == 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5':
            st.write("This chatbot is powered by Hugging Face's transformers library. It utilizes the Llama 2-13B language model, which is designed to provide intelligent and contextually relevant responses in natural language conversations.")
            st.write("Llama 2-13B is a large language model developed by a16z Infra. It is trained on diverse and extensive datasets to understand and generate human-like text. The model is capable of handling various types of conversational queries and providing informative responses.")
            # Display the additional image
            st.image("https://img.freepik.com/free-vector/cute-bot-say-users-hello-chatbot-greets-online-consultation_80328-195.jpg?w=740&t=st=1705818318~exp=1705818918~hmac=e1759ef1d2ab12f00c8f3b1aaf956e4913344e62ac5345bb094f0f059bb59559", caption="Hello from our Chatbot")
    else:
        # Display or clear chat messages
        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            with st.chat_message(latest_message["role"]):
                st.write(latest_message["content"])

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        if st.sidebar.button('Clear Chat History', on_click=clear_chat_history):
            pass  # This should be indented inside the if statement

        # User-provided prompt
        if option == "Llama 2":
            if st.button("Ask Question in Voice"):
                voice_input = get_voice_input()
                if voice_input:
                    st.session_state.messages.append({"role": "user", "content": voice_input})
                    with st.chat_message("user"):
                        st.write(f"User (Voice): {voice_input}")

                    # Generate a new response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = generate_llama2_response(voice_input, "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.", st.session_state.llm, st.session_state.temperature, st.session_state.top_p, st.session_state.max_length, st.session_state.messages)
                            placeholder = st.empty()
                            full_response = ''
                            for item in response:
                                full_response += item
                                placeholder.markdown(full_response)
                            placeholder.markdown(full_response)

                            # Play the response as speech (only if not already played)
                            if not st.session_state.audio_played:
                                play_text_as_speech(full_response)
                                st.session_state.audio_played = False

                            # Download the synthesized speech as audio

            # User-provided text input
            if prompt_input := st.chat_input(disabled=not replicate_api):
                st.session_state.messages.append({"role": "user", "content": prompt_input})
                with st.chat_message("user"):
                    st.write(prompt_input)

                # Generate a new response if the last message is not from the assistant
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = generate_llama2_response(prompt_input, "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.", st.session_state.llm, st.session_state.temperature, st.session_state.top_p, st.session_state.max_length, st.session_state.messages)
                            placeholder = st.empty()
                            full_response = ''
                            for item in response:
                                full_response += item
                                placeholder.markdown(full_response)
                            placeholder.markdown(full_response)

                            # Play the response as speech (only if not already played)
                            if not st.session_state.audio_played:
                                play_text_as_speech(full_response)
                                st.session_state.audio_played = False

                            # Download the synthesized speech as audio

# ... (rest of the code)

if __name__ == "__main__":
    main()

