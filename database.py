from pymongo import MongoClient

uri = "mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(uri)

    # Test connection
    client.admin.command("ping")
    print("MongoDB connected successfully!")

    # Create / use database
    db = client["vision_memory"]

    # Create / use collection
    people = db["people"]

    person = {
        "name": "Chandan",
        "glasses": True,
        "emotion": "happy",
        "age": "20-30"
    }

    result = people.insert_one(person)

    print("Inserted document ID:", result.inserted_id)

except Exception as e:
    print("Error:", e)

for person in people.find():
    
    print(person)