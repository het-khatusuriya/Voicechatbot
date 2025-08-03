# streamlit run chat.py


import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from better_profanity import Profanity
from google.generativeai import GenerativeModel
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr  # Added for voice input
import pyttsx3

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

pf = Profanity()


def text_to_speech(data):
    engine = pyttsx3.init()
    engine.say(data)
    engine.runAndWait()


def is_query_appropriate(query):
    """Check if the query contains inappropriate content using better-profanity"""
    return not pf.contains_profanity(query)

def initialize_file_chatbot(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embeddings)

    prompt_template = """
    You are an AI assistant with knowledge limited to the provided context. Use the following context to answer the question. If the question is unrelated to the context, respond with "Sorry, not much information available on that topic."

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def process_image_query(image_path, query):
    img = Image.open(image_path)
    model = GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([query, img])
    return response.text

# Added function to get voice input from microphone
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now")
        audio = recognizer.listen(source, timeout=5)
    return recognizer.recognize_google(audio)

def main():
    st.title("Multi-Modal Chatbot")
    st.write("Choose whether to process a file (PDF) or an image.")
    mode = st.selectbox("Select input type:", ["File (PDF)", "Image"])
    error_message = "Sorry, inappropriate query detected"
    if mode == "File (PDF)":
        file_source = st.radio("How would you like to provide the file?", ("Upload", "Enter Path"))

        if file_source == "Upload":
            uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
            if uploaded_file:
                file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        else:
            file_path = st.text_input("Enter the path to your PDF file:")

        if "file_path" in locals() and file_path:
            if (
                "qa_chain" not in st.session_state
                or st.session_state.get("file_path") != file_path
            ):
                with st.spinner("Initializing file chatbot..."):
                    st.session_state.qa_chain = initialize_file_chatbot(file_path)
                    st.session_state.file_path = file_path

            query = None  # Initialize query variable

            text_query = st.text_input("Enter your question about the PDF:")
            if text_query:
                query = text_query

            if st.button("Click to Speak"):
                try:
                    voice_input = get_voice_input()
                    st.success(f"You said: {voice_input}")
                    query = voice_input
                except:
                    st.error("Sorry, could not understand your voice.")

            if query:
                if is_query_appropriate(query):
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.qa_chain.invoke(query)
                        text_to_speech(answer["result"])
                        st.write("Answer:", answer["result"])
                else:
                    text_to_speech(error_message)
                    st.error(error_message)

    elif mode == "Image":
        image_source = st.radio("How would you like to provide the image?", ("Upload", "Enter Path"))

        if image_source == "Upload":
            uploaded_image = st.file_uploader(
                "Upload an image", type=["jpg", "png", "jpeg"]
            )
            if uploaded_image:
                image_path = os.path.join("temp", uploaded_image.name)
                os.makedirs("temp", exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                st.image(image_path, caption="Uploaded Image", use_column_width=True)
        else:
            image_path = st.text_input("Enter the path to your image:")

        if 'image_path' in locals() and image_path:
            query = None

            text_query = st.text_input("Enter your question about the image:")
            if text_query:
                query = text_query

            if st.button("Click to Speak"):
                try:
                    voice_input = get_voice_input()
                    st.success(f"You said: {voice_input}")
                    query = voice_input
                except:
                    st.error("Sorry, could not understand your voice.")

            if query:
                if is_query_appropriate(query):
                    with st.spinner("Generating answer..."):
                        answer = process_image_query(image_path, query)
                        text_to_speech(answer)
                        st.write("Answer:", answer)
                else:
                    text_to_speech(error_message)
                    st.error(error_message)
    if "tts_played" not in st.session_state:
        st.session_state.tts_played = True
        text_to_speech("Hi There!")
        text_to_speech("I am your multi-modal chatbot! here to help!")
        text_to_speech("Please Select input type:")


if __name__ == "__main__":
    main()
