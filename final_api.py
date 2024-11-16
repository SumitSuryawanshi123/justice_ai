from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
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
    case_details: str
    case_id: str
    case_title: str
    
class ChatRequest(BaseModel):
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

    

@app.post("/process_query/")
async def process_query(request: QueryRequest):
    try:
        # Extract query
        case_details = request.case_details
        case_id = request.case_id
        case_title = request.case_title
        
        # Get query embedding
        question_embedding = get_query_embedding(case_details)
        
        # Query Pinecone for similar texts
        query_result = index.query(vector=question_embedding, top_k=3, include_metadata=True)
        
        if "matches" not in query_result or len(query_result["matches"]) == 0:
            raise HTTPException(status_code=404, detail="No similar texts found in Pinecone index.")
        # Extract metadata from query result
        
        prev_case_facts = [
                    {
                        'case_details': texts[int(x["metadata"]['indexes'])],
                        'violated_articles': get_articals(labels[int(x["metadata"]['indexes'])])
                    }
                    for x in query_result["matches"]
                ]
        
        model_prompt_1 = f"I have case details as follow {case_details}"
        model_prompt_2 = f"You are my judge. Based on the context of similar previous cases, determine whether the person is guilty or not, and identify which articles are violated for the given case. Here are the similar previous cases and their outcomes: {prev_case_facts}"
        
        chat = model.start_chat(
            history=[
                {"role": "model", "parts": model_prompt_1+model_prompt_1 }
            ]
        )
        
        question = """ give me reponse in following format :
        {"guilty_or_not": "yes or no",
        "articles_violated": [],
        "points_of_violation":[],
        "comment" : ""}  if person is not guilty keep articles_violated and points_of_violation empty and if person is  guilty always include articles_violated with violatyed articals"""
        
        response = chat.send_message(question)
        
        insert_conversation(case_id = case_id, title = case_title, question = "initial prompt", answer = model_prompt_1+model_prompt_1)
        
        insert_conversation(case_id = case_id, title = case_title, question = question, answer = response.text)
        
        return  {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat_query/")
async def process_query(request: ChatRequest):
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
