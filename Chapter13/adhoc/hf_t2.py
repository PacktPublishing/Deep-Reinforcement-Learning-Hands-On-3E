from sentence_transformers import SentenceTransformer
import numpy as np

if __name__ == "__main__":
    c = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(c.get_sentence_embedding_dimension())
    r = c.encode(["I'm disappointed by delivery service", "Test sentence"], convert_to_tensor=True)
    print(r.shape)