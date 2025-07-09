import ast

with open("combine_tracks", "r") as f:
    
    print(f'song_name,artist_name,genre')
    for line in f:
        dct = line
        dct = ast.literal_eval(dct)
        song = dct["song_name"]
        artist = dct["artist_name"]
        genre = []
        st = ""
        for i in range(len(dct["tags"])):
            genre.append(dct["tags"][i][8:])
        for i in genre:
            st += i + ";"
        print(f'{song},{artist},\"{st}\"')
