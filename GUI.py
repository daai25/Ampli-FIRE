import streamlit as st

# ----- Page Config -----
st.set_page_config(page_title="Upload or Enter Text", layout="centered")

# ----- Custom CSS for Black Background -----
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

# ----- Upload Section -----
st.title("🎵 Upload or Enter Text")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "json"])

# ----- Textboxes with "OR" between them -----
col1, col2, col3 = st.columns([1, 0.2, 1])
with col1:
    text1 = st.text_input("Input Option 1")
with col2:
    st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)
with col3:
    text2 = st.text_input("Input Option 2")

# Add another "OR" below second input
st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Third textbox below
text3 = st.text_input("Input Option 3")
