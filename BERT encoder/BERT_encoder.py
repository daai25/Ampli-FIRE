from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
from tqdm import tqdm
import json

# Clear cache: rm -rf ~/.cache/huggingface/hub
# BERT-Modell laden
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('songs_data_test.json', 'r', encoding='utf-8') as f:
    songs = json.load(f)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    dbname="postgres",
    user="postgres",
    password="123",
    port=5432
)
cur = conn.cursor()

cur.execute("""Drop table if exists songs""")

# ✅ Erstelle Tabelle einmalig, falls sie nicht existiert
cur.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT NOT NULL,
        genre TEXT
    )
""")
conn.commit()

print("length of songs:", len(songs))
# 🔥 Prozessiere Songs und speichere in DB
#for song in tqdm(songs, desc="Songs verarbeiten"):
for song in songs:
    #print(f"Processing song: {song['song_name']} - {song['artist']} - {song['genre']}")
    title = song["song_name"]
    artist = song["artist"]
    genre = song["genre"]

    #print(f"Processing song: {title} - {artist} - {genre}")

    #input_text = f"{artist} - {title}"
    #embedding = model.encode(input_text).tolist()  # 384-dim List

    cur.execute(
        """
        INSERT INTO songs (title, artist, genre)
        VALUES (%s, %s, %s)
        """,
        (title, artist, genre)
    )


#cur.execute("SELECT * FROM songs")

#rows = cur.fetchall()





# ✅ Commit & Verbindung schließen nach dem Loop
conn.commit()
cur.close()
conn.close()

print("Alle Songs erfolgreich in PostgreSQL gespeichert!")