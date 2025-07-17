import psycopg2
import os
import json
from collections import defaultdict
from dotenv import load_dotenv

# Your original connection setup (unchanged)
load_dotenv(dotenv_path="database.env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

class GenreGrouper:
    def __init__(self, genre_json_path="genre_groups.json"):
        self.genre_to_group = self._load_genre_groups(genre_json_path)

    def _load_genre_groups(self, path):
        with open(path, "r") as f:
            genre_groups = json.load(f)
        mapping = {}
        for group, genres in genre_groups.items():
            for genre in genres:
                mapping[genre.lower()] = group
        return mapping

    def get_fingerprinted_song_groups(self):
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = """
        SELECT s.song_id, s.song_name, s.artist_name, g.genre_name
        FROM songs s
        JOIN song_genres sg ON s.song_id = sg.song_id
        JOIN genres g ON sg.genre_id = g.genre_id
        WHERE s.fingerprint IS NOT NULL AND s.fingerprint <> '';
        """
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = defaultdict(lambda: {
            "song_name": "",
            "artist_name": "",
            "genres": set(),
            "genre_groups": set()
        })

        for song_id, song_name, artist_name, genre_name in rows:
            key = f"{song_id}"
            result[key]["song_name"] = song_name
            result[key]["artist_name"] = artist_name
            result[key]["genres"].add(genre_name)
            group = self.genre_to_group.get(genre_name.lower(), "Uncategorized")
            result[key]["genre_groups"].add(group)

        return result

    def to_json(self):
        grouped_data = self.get_fingerprinted_song_groups()
        json_data = []

        for entry in grouped_data.values():
            json_data.append({
                "song_name": entry["song_name"],
                "artist_name": entry["artist_name"],
                "genres": sorted(list(entry["genres"])),
                "genre_groups": sorted(list(entry["genre_groups"]))
            })

        return json_data

    def save_json(self, filepath="fingerprinted_songs.json"):
        data = self.to_json()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved to {filepath}")

    def display(self):
        for entry in self.to_json():
            print(f"\n🎵 {entry['song_name']} by {entry['artist_name']}")
            print(f"  Genres: {', '.join(entry['genres'])}")
            print(f"  Genre Groups: {', '.join(entry['genre_groups'])}")
