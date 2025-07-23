import os
import csv
from collections import defaultdict
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="database.env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Genre grouping map
keyword_map = {
    "Rock": ["rock", "alternative rock", "indie rock", "garage"],
    "Pop": ["pop", "electropop", "dance pop"],
    "Hip-Hop & Rap": ["hip hop", "rap", "trap", "drill"],
    "Electronic": ["electronic", "edm", "techno", "house", "trance", "dubstep"],
    "R&B & Soul": ["r&b", "soul", "neo soul", "contemporary r&b"],
    "Jazz": ["jazz", "bebop", "smooth jazz", "fusion"],
    "Classical": ["classical", "orchestral", "baroque", "symphony"],
    "Country & Folk": ["country", "folk", "americana", "bluegrass"],
    "Latin": ["latin", "reggaeton", "salsa", "bachata", "cumbia"],
    "Metal": ["metal", "heavy metal", "black metal", "thrash", "death metal"],
    "Punk & Hardcore": ["punk", "hardcore", "emo", "post-punk"],
    "Reggae & Ska": ["reggae", "ska", "dub"],
    "World & International": ["afro", "world", "k-pop", "j-pop", "bhangra", "afrobeats"],
    "Blues": ["blues", "delta blues", "electric blues"],
    "Other": []  # fallback
}

# Helper to map genre to group
def map_genre_to_group(genre):
    genre_lower = genre.lower()
    for group, keywords in keyword_map.items():
        if any(keyword in genre_lower for keyword in keywords):
            return group
    return "Other"

# Connect to the PostgreSQL database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Query songs with their genres
cur.execute("""
    SELECT 
        s.song_name,
        s.artist_name,
        g.genre_name
    FROM songs s
    JOIN song_genres sg ON s.song_id = sg.song_id
    JOIN genres g ON sg.genre_id = g.genre_id
    WHERE s.song_name IS NOT NULL AND s.song_name <> ''
      AND s.artist_name IS NOT NULL AND s.artist_name <> ''
    ORDER BY s.song_name, s.artist_name, g.genre_name;
""")

rows = cur.fetchall()

# Group genre categories per song+artist
songs_dict = defaultdict(set)

for song_name, artist_name, genre_name in rows:
    key = (song_name, artist_name)
    genre_group = map_genre_to_group(genre_name)
    songs_dict[key].add(genre_group)

# Write to CSV file
with open('songs_with_genre_groups.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Song Name', 'Artist Name', 'Genre Groups'])

    for (song_name, artist_name), genre_groups in sorted(songs_dict.items()):
        writer.writerow([song_name, artist_name, ', '.join(sorted(genre_groups))])

# Close the connection
cur.close()
conn.close()
