from pymongo import MongoClient

uri = "mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri)

db = client["vision_memory"]

people = db["people"]


def save_person(name, glasses, emotion, age):

    person = {
        "name": name,
        "glasses": glasses,
        "emotion": emotion,
        "age": age
    }

    people.insert_one(person)


def get_person(name):

    return people.find_one({"name": name})