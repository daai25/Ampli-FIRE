import streamlit as st
import time

coll1, coll2, coll3 = st.columns([3, 2, 3])
with coll2:
    st.image("Amplifire_logo.png", width=150)

# Configuring Page
st.set_page_config(page_title="Upload or Enter Text", layout="centered")

st.session_state.step = 1

st.session_state.count = 0

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

uploaded_file = st.file_uploader("Upload a file", type=["mp3"], disabled=st.session_state.count is not 0)

st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Text boxes
# Disables text box if user uploads a file.
song_name = st.text_input("Song Name:", disabled=uploaded_file is not None)
if song_name:
    st.session_state.count = 1

col1, col2 = st.columns([1, 1])
with col1:
    artist_name = st.text_input("Artist Name:", disabled=uploaded_file is not None)
with col2:
    initial_genre = st.text_input("Genre:", disabled=uploaded_file is not None)

if st.session_state.step < 3:
    if st.button("Next"):
        # st.session_state.step += 1
        st.write("## Thinking...")
        st.write(" This should not take long. ")
        time.sleep(10)
        # Have if upload here
        # We translate the song into a spectrogram
        # We put it through the Deep Learning Model and the output goes into a genre variable
        # We put it through Cosine Similarity Model, output goes into variables
        # Have if text-based here
        # Put it through Linear Regression "Model"
        st.write("# Recommended Songs: ")
        # We display all the variables

# The Restart Button goes here...maybe
