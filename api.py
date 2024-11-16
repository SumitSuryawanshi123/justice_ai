from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from groq import Groq
import google.generativeai as genai
import pandas as pd
from db import insert_conversation, retrieve_conversation_by_case, check_case_id_exists

# Load environment variables
load_dotenv()
df = pd.read_csv('new_data_2.csv')

texts = df['facts'].tolist()
labels = df['labels'].to_list()
# Initialize API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("iit-index")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI application setup
app = FastAPI()

import json

# Path to the JSON file
file_path = './artical.json'

# Open and load the JSON file
with open(file_path, 'r',encoding='utf-8') as file:
    artical_data = json.load(file)


# Pydantic model for query input
class QueryRequest(BaseModel):
    user_query: str
    case_id : str

# Helper function to get query embeddings
def get_query_embedding(text):
    return embeddings.embed_query(text)

def get_articals(artical_list):
    elements = artical_list.strip("[]").split("' '")
    elements[0] = elements[0].lstrip("'")
    elements[-1] = elements[-1].rstrip("'")

    result = []
    for i in elements:
        result.append(artical_data.get(i, {}))
    
    return result

    

@app.post("/chat_query/")
async def process_query(request: QueryRequest):
    try:
        # Extract query
        case_id = request.case_id
        user_query = request.user_query
    

        chat = model.start_chat(
            history = retrieve_conversation_by_case(case_id)
        )
            
        response = chat.send_message(user_query)
        
        return  {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
