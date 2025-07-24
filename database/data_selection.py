import psycopg2
from dotenv import load_dotenv
import os

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

def __init__(self, env_path="database.env"):
    load_dotenv(dotenv_path=env_path)
    self.db_config = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }
    self.conn = psycopg2.connect(**self.db_config)
    self.cur = self.conn.cursor()

def get_all_songs(self):
    self.cur.execute("SELECT song_id, song_name, artist_name FROM songs ORDER BY artist_name;")
    return self.cur.fetchall()

def get_songs_by_artist(self, artist_name):
    self.cur.execute("""
        SELECT song_id, song_name FROM songs
        WHERE artist_name = %s
        ORDER BY song_name;
    """, (artist_name.strip().lower(),))
    return self.cur.fetchall()

def get_songs_by_genre(self, genre_name):
    self.cur.execute("""
        SELECT s.song_id, s.song_name, s.artist_name
            FROM songs s
            JOIN song_genres sg ON s.song_id = sg.song_id
            JOIN genres g ON sg.genre_id = g.genre_id
            WHERE g.genre_name = %s
            ORDER BY s.song_name;
        """, (genre_name.strip().lower(),))
        return self.cur.fetchall()

    def get_genres_of_song(self, song_name, artist_name):
        self.cur.execute("""
            SELECT g.genre_name
            FROM genres g
            JOIN song_genres sg ON g.genre_id = sg.genre_id
            JOIN songs s ON s.song_id = sg.song_id
            WHERE s.song_name = %s AND s.artist_name = %s;
        """, (song_name.strip().lower(), artist_name.strip().lower()))
        return [row[0] for row in self.cur.fetchall()]

    def close(self):
        self.cur.close()
        self.conn.close()