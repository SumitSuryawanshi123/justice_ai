from pymongo import MongoClient
from datetime import datetime

mongo_uri = "mongodb+srv://mkv:mkv09@cluster0.kgq1cs6.mongodb.net/"
client = MongoClient(mongo_uri)

db = client["legal_chatbot"]

collection = db["conversation_history"]

# Function to insert a record into the collection
def transform_conversation(conversation_list):
    result = []
    
    for item in conversation_list:
        # Append the model's response first
        result.append({
            "role": "model",
            "parts": f"'{item['model']}'"
        })
        
        # Append the user's question second
        result.append({
            "role": "user",
            "parts": item['user']
        })
    
    return result


def insert_conversation(case_id, title, question, answer):
    current_time = datetime.now()
    
    document = {
        "case_id": case_id,
        "case_title": title,
        "user": question,
        "model": answer,
        "date": current_time.date().isoformat(),  # Only the date part (YYYY-MM-DD)
        "time": current_time.time().isoformat()   # Only the time part (HH:MM:SS)
    }

    # Insert the document into the collection
    result = collection.insert_one(document)
    
    # Output the inserted ID

def get_first_record(case_id):
    # Query to find the first (oldest) record for the given case_id
    cursor = collection.find(
        {"case_id": case_id},  # Match the case_id
        {"_id": 0, "user": 1, "model": 1}  # Include 'user', 'model', 'date', and 'time' fields
    ).sort([("date", 1), ("time", 1)])  # Sort by 'date' and 'time' in ascending order (oldest first)
    
    # Limit the result to one record
    record = cursor.limit(1).next()  # Use .next() to fetch the first record
    
    return transform_conversation(record)

def check_case_id_exists(case_id):
    # Query to check if the given case_id exists in the collection
    result = collection.find_one({"case_id": case_id})
    
    if result:
        return True  # case_id found
    else:
        return False  # case_id not found

# Function to retrieve conversation history based on case_id, sorted by date and time
def retrieve_conversation_by_case(case_id):
    # Sort by 'date' and 'time', and only return the 'user' and 'model' fields
    cursor = collection.find(
        {"case_id": case_id},  # Match the case_id
        {"_id": 0, "user": 1, "model": 1}  # Include only 'user' and 'model' fields, exclude '_id'
    ).sort([("date", 1), ("time", 1)])  # Sort by 'date' and 'time' in ascending order

    result = []
    
    for doc in cursor:
        result.append(doc)
        
    return transform_conversation(result)

# # Example usage of the functions
# insert_conversation(
#     "abc123", 
#     "Sample Conversation", 
#     "What is MongoDB?", 
#     "MongoDB is a NoSQL database used to store data in a flexible, JSON-like format. srgf24regrg"
# )

# # Retrieve and print the conversation for a specific case_id
# print(get_first_record("abc123"))
