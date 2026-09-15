from main import BaseMemoryStrategy
import faiss
import numpy as np
from main import generate_embedding


class RetrievalMemory(BaseMemoryStrategy):
    def __init__(self, k: int = 3, embedding_dim: int = 4096):
        self.k = k
        self.embedding_dim = embedding_dim

        self.document = []
        # Initialize a FAISS index. IndexFlatL2 performs an exhaustive search using
        # L2 (Euclidean) distance, which is effective for a moderate number of vectors.
        self.index = faiss.IndexFlat(self.embedding_dim)

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new conversational turn to the memory. Each part of the turn (user
        input and AI response) is embedded and indexed separately for granular retrieval.
        """
        # We store each part of the turn as a separate document to allow for more
        # precise matching. For example, a query might be similar to a past user
        # statement but not the AI's response in that same turn.
        doc_to_add = [
            f"User ask: {user_input}",
            f"AI response: {ai_response}",
        ]

        for document in doc_to_add:
            embedding = generate_embedding(document)
            if embedding:
                # Store the original text. The index of this document will correspond
                # to the index of its vector in the FAISS index.
                self.document.append(document)
                print(len(embedding))
                vec = np.array([embedding], dtype='float32')
                self.index.add(vec)

    def get_context(self, query: str) -> str:
        if self.index.ntotal == 0:
            return "No information in memory yet"

        query_embedding = generate_embedding(query)

        if not query_embedding:
            return "Can not process query for retrieval"

        query_vec = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_vec, self.k)

        retrieval = [self.document[i] for i in indices[0] if i != -1]

        if not retrieval:
            return "Can not find any relevant information"

        return "### Relevant Information Retrieved from Memory:\n" + "\n---\n".join(retrieval)

    def clear(self):
        self.document = []
