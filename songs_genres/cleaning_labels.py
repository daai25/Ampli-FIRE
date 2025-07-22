import csv

# Define the genre mapping
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
    "Other": []
}

def map_genre(genre_str):
    genre_str_lower = genre_str.lower()
    for group, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in genre_str_lower:
                return group
    return "Other"

# Read and process the file
input_file = "labeled_song_artists"
output_file = "labeled_song_artists_grouped.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)  # write header as-is

    for row in reader:
        if len(row) >= 4:
            row[-1] = map_genre(row[-1])
        writer.writerow(row)

print(f"Genre grouping complete. Output written to '{output_file}'")