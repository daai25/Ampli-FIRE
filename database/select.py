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

def select_songs(songsong_name, song_artist):
    song_name = song_name.strip().lower()
    song_artist = song_artist.strip().lower()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        # Get song_id for the given song
        cur.execute("""
            SELECT song_id FROM songs
            WHERE song_name = %s AND artist_name = %s;
        """, (song_name, song_artist))
        result = cur.fetchone()

        if not result:
            print("Song not found.")
            return

        song_id = result[0]

        # Delete from song_genres_db first
        cur.execute("DELETE FROM song_genres WHERE song_id = %s;", (song_id,))
        # Then delete the song from song_db
        cur.execute("DELETE FROM songs WHERE song_id = %s;", (song_id,))

        print(f"Deleted '{song_name}' by '{song_artist}' (ID: {song_id})")

    except Exception as e:
        print(f"Error deleting song: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()