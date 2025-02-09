import os

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

try:
    conn = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = conn.get_database()

    # Test the connection
    conn.admin.command("ping")
    print("Connected to MongoDB successfully!")
except ConnectionFailure as e:
    print(f"MongoDB connection failed: {e}")
    exit(1)
