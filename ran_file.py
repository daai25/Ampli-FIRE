import psycopg2

conn= psycopg2.connect(
    host = "galdurs-laptop.tail83824f.ts.net",
    port=5433,
    user="song_writer",
    password="a11Bk",
    database="music_db"
        )

print("Connected Sucessfully!")
