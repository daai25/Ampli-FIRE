import json
import csv

with open("music_data_readable.json") as file1:

    json_data = json.load(file1)

json_song = {entry["song_name"]: entry for entry in json_data}

music_lst = []

with open("cleaned_music_jamendo.csv") as file2:
    
    reading = csv.DictReader(file2)
    for row in reading:
        csv_song = row["song_name"]
        csv_artist = row["artist_name"]
        csv_genre = row["genre"]
        if csv_song in json_song:
            json_artist = json_song[csv_song]["artist"]
            if csv_artist == json_artist:
                json_genre = json_song[csv_song]["genre"]
                if json_genre in csv_genre:
                    json_genre = csv_genre
                    msuic_lst.append({
                    'song': csv_song,
                    'artist': csv_artist,
                    'genre': json_genre})
                elif json_genre not in csv_genre:
                    csv_genre += f'{json_genre}'
                    json_genre = csv_genre.split(";")
                    music_lst.append({
                    'song': csv_song,
                    'artist': csv_artist, 
                    'genre': json_genre})
            else:
                continue
        else:
            continue

print(music_lst)


with open("music_dump.json", "w") as file:
    json.dump(music_lst, file, indent=4)

