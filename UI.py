import streamlit as st

st.markdown(
    """
    <div style="text-align: center;">
        <img src="Amplifire_logo.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# Configuring Page
st.set_page_config(page_title="Upload or Enter Text", layout="centered")

st.session_state.step = 1

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
text1 = st.text_input("Song Name:")

col1, col2 = st.columns([1, 1])
with col1:
    text2 = st.text_input("Artist Name:")
with col2:
    text3 = st.text_input("Genre:")

if st.session_state.step < 3:
    if st.button("Next"):
        st.session_state.step += 1


