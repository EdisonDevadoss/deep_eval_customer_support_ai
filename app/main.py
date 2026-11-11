from fastapi import FastAPI
from app.chatbot import ask_customer_support

app = FastAPI()


@app.get("/chat")
def chat(question: str):
    return ask_customer_support(question)
