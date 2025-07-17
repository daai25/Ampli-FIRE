import psycopg2
from dotenv import load_dotenv
import os
import json
import csv

# Load the specific .env file for database connection
load_dotenv(dotenv_path="database.env")

# Get the environment variables 
# from a .env so that it will be ignored in github
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

#insert file_path to the file you want to extract the songs list from
file_path = "scraping_jamendo\cleaned_music_jamendo.csv"  # or "your_input_file.csv"

#extract the objects from the CSV file and put it in a list
def load_songs_from_csv(csv_path):
    songs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_genres = row.get("genre", "").strip().lower()

            # Normalize genres: if genres is just a string, wrap it in a list
            if ";" in raw_genres:
                genres = [g.strip().lower() for g in raw_genres.split(";") if g.strip()]
            elif raw_genres:
                genres = [raw_genres.strip().lower()]
            else:
                genres = []

            songs.append({
                "song_name": row["song_name"].strip().lower(),
                "artist_name": row["artist_name"].strip().lower(),
                "genres": genres
            })
    return songs

#extract the objects from the JSON file and put it in a list
def load_songs_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        songs = []
        for song in raw_data:
            genres = song.get("genre", [])

            # Normalize: if genres is just a string, wrap it in a list
            if isinstance(genres, str):
                genres = [genres.strip().lower()]
            elif isinstance(genres, list):
                genres = [g.strip().lower() for g in genres if g.strip()]
            else:
                genres = []

            songs.append({
                "song_name": song["song_name"].strip().lower(),
                "artist_name": song["artist_name"].strip().lower(),
                "genres": genres
            })
        return songs

#insert songs into the database
def insert_songs(songs_data):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    for song in songs_data:
        song_name = song["song_name"]
        artist_name = song["artist_name"]
        genres = song.get("genres", [])

        if not genres:
            print(f"Skipping '{song_name}' — there has to be at least a genre for a song.")
            continue

        # Insert song
        cur.execute("""
            INSERT INTO songs (song_name, artist_name)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            RETURNING song_id;
        """, (song_name, artist_name))

        result = cur.fetchone()
        if result:
            song_id = result[0]
        else:
            # Song already exists, fetch its ID
            cur.execute("""
                SELECT song_id FROM songs WHERE song_name = %s AND artist_name = %s;
            """, (song_name, artist_name))
            song_id = cur.fetchone()[0]

        for genre_name in genres:
            genre_name = genre_name.strip().lower()

            # Insert genre if not exists
            cur.execute("""
                INSERT INTO genres (genre_name)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, (genre_name,))

            # Get genre_id
            cur.execute("""
                SELECT genre_id FROM genres WHERE genre_name = %s;
            """, (genre_name,))
            genre_id = cur.fetchone()[0]

            # Link song to genre
            cur.execute("""
                INSERT INTO song_genres (song_id, genre_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
            """, (song_id, genre_id))

    cur.close()
    conn.close()
    print("All songs inserted successfully.")

# Load from CSV or JSON, if other filetypereturn an errormessage
try:
    if file_path.endswith(".json"):
        songs_data = load_songs_from_json(file_path)
    elif file_path.endswith(".csv"):
        songs_data = load_songs_from_csv(file_path)
    else:
        raise ValueError("Unsupported file type — must be .json or .csv")

    insert_songs(songs_data)

#catch exceptions
except Exception as e:
    print(f"Something went wrong during file load or insertion: {e}")
