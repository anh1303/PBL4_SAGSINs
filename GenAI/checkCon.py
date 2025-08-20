from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Replace <password> with your actual password
uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Pinged your deployment. Connected to MongoDB Atlas successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
