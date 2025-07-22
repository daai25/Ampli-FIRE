import pandas as pd

# Load the CSV file
input_file = 'deduplicated_songs.csv'  # Replace with your actual file name
output_file = 'cleaned_songs.csv'      # Output file name

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Drop rows where songname or artistname is missing (NaN or empty string)
df = df.dropna(subset=['song_name', 'artist_name'])
df = df[(df['song_name'].str.strip() != '') & (df['artist_name'].str.strip() != '')]

# Drop duplicates based on songname and artistname
df = df.drop_duplicates(subset=['song_name', 'artist_name'])

# Save to a new CSV file
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to '{output_file}'")