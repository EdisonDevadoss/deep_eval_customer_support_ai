from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent / "knowledge_base"

documents = [file.read_text() for file in BASE_DIR.glob("*.txt")]

vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents)


def retrieve_documents(question, top_k=2):
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(question_vector, document_vectors)[0]

    top_indices = scores.argsort()[::-1][:top_k]

    return [documents[i] for i in top_indices]
