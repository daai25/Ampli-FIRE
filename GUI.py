import streamlit as st

# Configuring Page
st.set_page_config(page_title="Upload or Enter Text", layout="centered")

# Background as black
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #222;
        color: white;
    }
    .stFileUploader>div>div {
        color: white;
    }
    .or-divider {
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Upload section
st.title("Upload or Enter Text")

uploaded_file = st.file_uploader("Upload a file", type=["mp3"])

# Text boxes
col1, col2, col3 = st.columns([1, 0.2, 1])
with col1:
    text1 = st.text_input("Song Name:")
with col2:
    text2 = st.text_input("Artist Name:")
with col3:
    text3 = st.text_input("Genre:")