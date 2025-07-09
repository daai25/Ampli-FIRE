#!/usr/bin/env python3

import csv
import sys
import os

tracks = {}

with open('cleaning_raw_meta.csv', newline='\n') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for row in reader:
        # TRACK_ID,TRACK_NAME,ARTIST_NAME
        track = { "id": row[0], "song_name": row[1], "artist_name": row[2] }
        tracks[track["id"]] = track

with open('clean_data_genre.csv', newline='\n') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for row in reader:
        track_id = row[0]
        tags = row[1:]
        track_tags = tags
        tracks[track_id]["tags"] = track_tags
        print(tracks[track_id])
    
#cleaning_raw_meta.csv
