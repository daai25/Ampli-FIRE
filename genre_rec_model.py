import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# 1. Load dataset from CSV
# Make sure 'songs.csv' has columns: song, artist, genre
df = pd.read_csv("cleaned_songs.csv")

# 2. Encode genre as numeric
le = LabelEncoder()
df['genre_num'] = le.fit_transform(df['genres'])

# 3. Simulate a feature and train dummy model
df['dummy_feature'] = range(len(df))
X = df[['dummy_feature']]
y = df['genre_num']

model = LinearRegression()
model.fit(X, y)


# 4. Recommendation function (based on genre match)
def recommend_by_genre(genre, n_pool=100, top_n=4):

    # 1. Filter songs by genre
    genre_songs = df[df['genres'] == genre]

    # 2. Take the first n_pool songs (e.g., 100)
    top_n_pool = genre_songs.head(n_pool)

    # 3. Shuffle the top_n_pool
    shuffled = top_n_pool.sample(frac=1).reset_index(drop=True)

    # 4. Return the first n_return songs (e.g., 4)
    return shuffled[['song_name', 'artist_name', 'genres']].head(top_n)


