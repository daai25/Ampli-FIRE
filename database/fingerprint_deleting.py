import psycopg2
from dotenv import load_dotenv
import os

# Load the specific .env file for database connection
load_dotenv(dotenv_path="database.env")

# Get the environment variables 
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def remove_fingerprints():
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Update fingerprints to NULL
        update_query = "UPDATE songs SET fingerprint = NULL;"
        cur.execute(update_query)

        # Commit changes
        conn.commit()
        print("Fingerprints removed successfully.")

        # Clean up
        cur.close()
        conn.close()
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    remove_fingerprints()
