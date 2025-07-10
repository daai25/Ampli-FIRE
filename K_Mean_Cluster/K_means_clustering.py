from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
from sklearn.cluster import KMeans


# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    dbname="postgres",
    user="postgres",
    password="123",
    port=5432
)
cur = conn.cursor()

# Load BERT embdeddings from the database
cur.execute("SELECT id, bert_title, bert_artist, bert_genre FROM songs")
rows = cur.fetchall()

# for row in rows:
#     print(f"ID: {row[0]}, BERT Title: {row[1][:5]}...")  # Print first 5 dimensions for brevity

# Extract only the embedding columns (bert_title, bert_artist, bert_genre) and convert to float
X = []
for row in rows:
    col = row[3]  # Only bert_genre
    if isinstance(col, list) or isinstance(col, np.ndarray):
        embedding = col
    else:
        embedding = [float(x) for x in col.strip('{}').split(',')]
    X.append(embedding)
X = np.array(X, dtype=float)


# print(f"X shape: {X.shape}")  # Should be (number_of_songs, 1152) if each embedding is 384-dim for title, artist, and genre
# print(f"X: {X[:1]}")  # Print first embedding for brevity

n_Clusters = 3
kmeans = KMeans(n_clusters=n_Clusters, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
print("Cluster-Labels:", labels)






