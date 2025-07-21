import psycopg2
from dotenv import load_dotenv
import os
from pathlib import Path
import re

# Load environment variables
load_dotenv(dotenv_path="database.env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# === CONFIG: Set the folder path containing PNG files ===
FOLDER_PATH = "spectrograms_32"  # Change this to your folder
# ========================================================

def extract_song_and_artist(filename):
    """Extract song and artist names, normalize by replacing _ with space, lowercase, and trimming."""
    stem = Path(filename).stem

    # Remove trailing '_spectrogram' if present
    if stem.lower().endswith('_spectrogram'):
        stem = stem[: -len('_spectrogram')]

    # Match pattern: song_name - artist_name
    match = re.match(r"(.+?)\s*-\s*(.+)", stem)
    if not match:
        raise ValueError(f"Invalid filename format: '{filename}' (expected: song - artist[_spectrogram].png)")

    song_name = match.group(1).replace("_", " ").strip().lower()
    artist_name = match.group(2).replace("_", " ").strip().lower()
    return song_name, artist_name

def update_fingerprint(cursor, conn, image_path, song_name, artist_name):
    """Update the fingerprint if the song exists and has no fingerprint yet."""
    cursor.execute("""
        SELECT song_id, fingerprint FROM songs
        WHERE song_name = %s AND artist_name = %s
    """, (song_name, artist_name))
    result = cursor.fetchone()

    if result is None:
        print(f"Song not found: '{song_name}' by '{artist_name}'")
        return

    song_id, fingerprint = result
    if fingerprint is not None:
        print(f"Already has fingerprint: '{song_name}' by '{artist_name}'")
        return

    with open(image_path, "rb") as img_file:
        binary_data = img_file.read()

    cursor.execute("""
        UPDATE songs SET fingerprint = %s WHERE song_id = %s
    """, (psycopg2.Binary(binary_data), song_id))
    conn.commit()
    print(f"Inserted fingerprint: '{song_name}' by '{artist_name}'")

def process_folder(folder_path):
    """Go through each PNG file and insert fingerprint if necessary."""
    png_files = list(Path(folder_path).glob("*.png"))

    if not png_files:
        print("No PNG files found in the folder.")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for file_path in png_files:
            try:
                song_name, artist_name = extract_song_and_artist(file_path.name)
                update_fingerprint(cursor, conn, file_path, song_name, artist_name)
            except ValueError as ve:
                print(f"Skipping file '{file_path.name}': {ve}")
            except Exception as e:
                print(f"Error processing '{file_path.name}': {e}")

    except Exception as e:
        print(f"Database connection error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    process_folder(FOLDER_PATH)
