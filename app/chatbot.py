import os
from dotenv import load_dotenv
from google import genai

from app.retriever import retrieve_documents

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = os.getenv("GEMINI_MODEL")


def ask_customer_support(question):

    # 1. Retrieve relevant documents
    context = retrieve_documents(question)

    # 2. Prepare the prompt
    prompt = f"""
    You are a customer support assistant.

    Answer the customer's question using only
    the information in the provided context.

    If the answer is not available in the context,
    clearly say that you don't have enough information.

    Context:
    {context}

    Customer question:
    {question}

    Answer:
    """

    # 3. Call Gemini
    response = client.models.generate_content(model=MODEL, contents=prompt)

    # 4. Return answer and retrieved context
    return {"question": question, "answer": response.text, "retrieval_context": context}
