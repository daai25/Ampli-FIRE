import streamlit as st
import time
from genre_rec_model import recommend_by_genre

coll1, coll2, coll3 = st.columns([3, 2, 3])
with coll2:
    st.image("Amplifire_logo.png", width=150)

# Configuring Page
st.set_page_config(page_title="Ampli-FIRE", layout="centered")

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'show_recommend' not in st.session_state:
    st.session_state.show_recommend = False

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
    /* Make button text black */
    div.stButton > button {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Upload section
st.title("Upload or Enter Text")

uploaded_file = st.file_uploader("Upload a file", type=["mp3"], disabled=st.session_state.count is not 0)

st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)

# Text boxes
# Disables text box if user uploads a file.

genre_options = ["Rock", "Pop", "Hip-Hop & Rap", "Electronic", "R&B & Soul", "Jazz", "Classical",
                 "Country & Folk", "Latin", "Metal", "Punk & Hardcore", "Reggae & Ska",
                 "World & International", "Blues", "Other"]

song_name = st.text_input("Song Name:", disabled=uploaded_file is not None)
if song_name:
    st.session_state.count = 1

col1, col2 = st.columns([1, 1])
with col1:
    artist_name = st.text_input("Artist Name:", disabled=uploaded_file is not None)
with col2:
    initial_genre = st.selectbox("Genre:",genre_options, disabled=uploaded_file is not None)

if st.session_state.step < 3:
    if st.button("Next"):
        st.session_state.show_recommend = True
        st.write("# Recommended Songs: ")
        if uploaded_file is None:
            st.write(recommend_by_genre(initial_genre))

if st.session_state.show_recommend:
    if st.button("Recommend Again"):
        st.write("# Recommended Songs: ")
        st.write(recommend_by_genre(initial_genre))