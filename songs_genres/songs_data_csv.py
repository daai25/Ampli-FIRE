import os
import csv
import json
from collections import defaultdict

# === CONFIGURATION ===
input_folder = "database\input_files"
output_file = "deduplicated_songs.csv"

# === KEYWORD MAP (GROUPING BASED ON GENRE FRAGMENTS) ===
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
    "Other": []  # fallback group
}

# === DEDUPLICATED SONG STORAGE ===
songs = defaultdict(set)

# === MAP GENRE TO GROUP BASED ON KEYWORDS ===
def map_to_group(genre):
    genre_lower = genre.strip().lower()
    for group, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in genre_lower:
                return group
    return "Other"

# === PROCESS EACH FILE ===
def process_file(filepath):
    _, ext = os.path.splitext(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        if ext.lower() == ".csv":
            reader = csv.DictReader(f)
            for row in reader:
                song = row.get("song_name", "").strip()
                artist = row.get("artist_name", "").strip()
                genres_raw = row.get("genre", row.get("genres", "")).strip().strip('"')
                genres = [g for g in genres_raw.split(";") if g]
                key = (song, artist)
                songs[key].update(map_to_group(g) for g in genres)

        elif ext.lower() == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("songs", data)

            for item in data:
                song = item.get("song_name") or item.get("song") or ""
                artist = item.get("artist", "")
                genre_field = item.get("genre") or item.get("genres", [])

                if isinstance(genre_field, str):
                    genres = [genre_field]
                elif isinstance(genre_field, list):
                    genres = genre_field
                else:
                    genres = []

                key = (song.strip(), artist.strip())
                songs[key].update(map_to_group(g) for g in genres if g)

# === PROCESS ALL INPUT FILES ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") or filename.endswith(".json"):
        process_file(os.path.join(input_folder, filename))

# === WRITE FINAL DEDUPLICATED CSV ===
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["song_name", "artist_name", "genres"])
    for (song, artist), genre_groups in sorted(songs.items()):
        genre_string = ";".join(sorted(set(genre_groups)))
        writer.writerow([song, artist, genre_string])

print(f"✅ Done! Grouped genre list saved to: {output_file}")
