import psycopg2
from dotenv import load_dotenv
import os
from collections import defaultdict
import json

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

class GenreGrouper:
    def __init__(self, db_config, table_name="genres", column_name="genre_name"):
        self.db_config = db_config
        self.table_name = table_name
        self.column_name = column_name
        self.genres = []
        self.groups = defaultdict(list)

    def fetch_genres(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    query = f"SELECT DISTINCT {self.column_name} FROM {self.table_name};"
                    cur.execute(query)
                    self.genres = [row[0] for row in cur.fetchall() if row[0]]
        except Exception as e:
            print("Error fetching genres:", e)

    def group_genres(self):
        # 15 total groups, including "Other"
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

        for genre in self.genres:
            lower_genre = genre.lower()
            matched = False
            for group, keywords in keyword_map.items():
                if any(keyword in lower_genre for keyword in keywords):
                    self.groups[group].append(genre)
                    matched = True
                    break
            if not matched:
                self.groups["Other"].append(genre)

    def print_groups(self):
        for group, items in self.groups.items():
            print(f"\n{group} ({len(items)}):")
            for item in items:
                print(f"  - {item}")

    def save_groups_to_json(self, filename="genre_groups.json"):
        try:
            # Convert defaultdict to regular dict for JSON compatibility
            output_data = {group: genres for group, genres in self.groups.items()}
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"\n Genre groups saved to {filename}")
        except Exception as e:
            print("Error writing to JSON:", e)

# Example usage
if __name__ == "__main__":
    grouper = GenreGrouper(DB_CONFIG)
    grouper.fetch_genres()
    grouper.group_genres()
    grouper.print_groups()
    grouper.save_groups_to_json("genre_groups.json")  # Save to JSON file
