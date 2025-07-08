import json
import os

#Path to the json with the links and genre of the songs
folder = "YOUR PATH WHERE YOU WANT TO SAVE YOUR FILE IN AT THE END"

#name of the json file, that the scraped data would be saved to
filename = os.path.join(folder, "NAME OF THE FILE")

# Beispiel: So würde man eine JSON-Datei laden
with open("web_scraping_result\everynoise_scraper\everynoise_scraper\music_data.json", "r", encoding="utf-8") as f:
    music_data = json.load(f)


# Delete "E/n" in Artist names
for song in music_data:
    song["artist"] = song["artist"].replace("E\n", "")


# Save properly encoded JSON with readable characters
with open(filename, "w", encoding="utf-8") as file: #filename or "FILE NAME YOU WANT IT TO BE CALLED"
    json.dump(music_data, file, indent=4, ensure_ascii=False)