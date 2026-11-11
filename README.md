# Customer Support AI

A small retrieval-augmented generation (RAG) customer-support chatbot that answers questions **only** from company policy documents, plus an evaluation suite built with [DeepEval](https://github.com/confident-ai/deepeval).

The bot retrieves the relevant policy text with TF-IDF, asks Google Gemini to answer from that text, and returns the answer together with the retrieved context. DeepEval then scores those answers for faithfulness, relevancy and policy compliance.

## How it works

```
question
   │
   ▼
retrieve_documents()      app/retriever.py   TF-IDF + cosine similarity, top 2 policy docs
   │
   ▼
ask_customer_support()    app/chatbot.py     prompt Gemini: "answer only from this context"
   │
   ▼
{ question, answer, retrieval_context }
   │
   ▼
GET /chat?question=...    app/main.py        FastAPI endpoint
```

- **Retriever** ([app/retriever.py](app/retriever.py)): loads every `.txt` file in `app/knowledge_base/` once, at import time, and builds a TF-IDF index. For each question it returns the 2 most similar documents.
- **Chatbot** ([app/chatbot.py](app/chatbot.py)): puts the retrieved documents into a prompt that tells the model to answer from that context only, and to say so when the context doesn't contain the answer.
- **API** ([app/main.py](app/main.py)): exposes the chatbot through a single `GET /chat` endpoint.

## Project structure

```
customer-support-ai/
├── app/
│   ├── main.py                  # FastAPI app (GET /chat)
│   ├── chatbot.py               # Prompt building + Gemini call
│   ├── retriever.py             # TF-IDF document retrieval
│   └── knowledge_base/
│       ├── return_policy.txt
│       ├── shipping_policy.txt
│       └── cancellation_policy.txt
├── tests/
│   ├── test_cases.py            # Evaluation inputs + expected behaviour
│   └── test_chatbot.py          # DeepEval metrics run via pytest
├── requirements.txt
└── .env                         # API key + model name (not committed)
```

## Tech stack

- Python 3.13
- FastAPI + Uvicorn
- Google Gen AI SDK (`google-genai`), using Gemini models
- scikit-learn (TF-IDF retrieval)
- DeepEval + pytest (evaluation)

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv eval_env
   source eval_env/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```env
   GOOGLE_API_KEY=your-google-api-key
   GEMINI_MODEL=gemini-2.5-flash
   ```

   `GEMINI_MODEL` is used both by the chatbot and as the judge model for the evaluations. `.env` is listed in `.gitignore`, so it won't be committed.

## Running the API

```bash
uvicorn app.main:app --reload
```

Ask a question:

```bash
curl "http://127.0.0.1:8000/chat?question=How%20long%20does%20shipping%20take%3F"
```

Example response:

```json
{
  "question": "How long does shipping take?",
  "answer": "Standard delivery takes 3-5 business days, and express delivery takes 1-2 business days.",
  "retrieval_context": ["Shipping Policy\n\n1. Standard delivery takes 3-5 business days. ...", "..."]
}
```

FastAPI also serves interactive docs at http://127.0.0.1:8000/docs.

## Evaluation

[tests/test_chatbot.py](tests/test_chatbot.py) runs every case in [tests/test_cases.py](tests/test_cases.py) through the chatbot and scores the result with three DeepEval metrics. Gemini is the judge model (`deepeval.models.GeminiModel`).

| Metric | What it checks | Threshold |
|---|---|---|
| `FaithfulnessMetric` | The answer is supported by the retrieved context (no hallucination) | 0.7 |
| `AnswerRelevancyMetric` | The answer actually addresses the question | 0.7 |
| `GEval` – "Policy Compliance" | The answer doesn't invent policies, promise unlisted exceptions, or contradict the policy | 0.7 |

### Test cases

| Input | Expected behaviour |
|---|---|
| Can I return a product after 20 days? | Explain the 30-day return policy. |
| Can I return a product after 45 days? | Explain that the return period is 30 days. |
| How long does shipping take? | Explain standard and express delivery times. |
| Can I cancel an order after shipping? | Explain that shipped orders cannot be cancelled. |
| Do you sell mobile phones? | Do not invent product information (out-of-scope check). |

### Running the evaluations

With pytest:

```bash
pytest tests/test_chatbot.py -v
```

Or with the DeepEval CLI, which prints a summary of the metric scores:

```bash
deepeval test run tests/test_chatbot.py
```

Each test case makes one call to generate the answer and several judge calls per metric, so a full run makes many Gemini requests. Watch your API quota and rate limits.

## Extending the project

- **Add a policy:** drop a new `.txt` file into `app/knowledge_base/`. It is indexed automatically the next time the app or tests start.
- **Add a test case:** append a dict with `input` and `expected_behavior` to `test_cases` in [tests/test_cases.py](tests/test_cases.py).
- **Add a metric:** create another DeepEval metric in [tests/test_chatbot.py](tests/test_chatbot.py) and add it to the list passed to `assert_test`.

## Limitations

- Retrieval is keyword-based TF-IDF, so it can miss questions that use different wording from the policy text.
- The retriever always returns the top 2 documents, even when neither is relevant. The prompt relies on the model to say it doesn't have enough information.
- The knowledge base is loaded once at startup. Restart the server after changing policy files.
