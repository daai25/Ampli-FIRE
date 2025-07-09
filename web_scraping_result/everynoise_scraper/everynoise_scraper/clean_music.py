import json

with open("music_data.json", "r") as file:
    data = json.load(file)

for song in data:
    song["song_name"] 
