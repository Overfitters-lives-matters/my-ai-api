from fastapi import FastAPI
from pydantic import BaseModel
from src.llm_client import LLMClient
from src.conversation import Conversation
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

conversation = Conversation()
client = LLMClient(api_key=os.getenv("OPENROUTER_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation.add_user_message(request.message)
    response = client.stream_response(conversation.get_history())
    conversation.add_assistant_message(response)
    return ChatResponse(response=response)
@app.get("/history")
async def get_history():
    return {"history": conversation.get_history()}
@app.delete("/history")
async def clear_history():
    conversation.messages = []
    return {"message": "conversation cleared"}