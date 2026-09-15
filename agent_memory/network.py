from main import BaseMemoryStrategy
import networkx
from main import generate_text
import re


class GraphMemory(BaseMemoryStrategy):
    def __init__(self):
        self.graph = networkx.DiGraph()

    def _extract_triples(self, text: str) -> list(tuple[str, str, str]):
        """
        Uses the LLM to extract knowledge triples (Subject, Relation, Object) from a given text.
        This is a form of "LLM as a Tool" where the model's language understanding is
        used to create structured data.
        """
        print("--- [Graph Memory: Attempting to extract triples from text.] ---")
        # Construct a detailed prompt that instructs the LLM on its role and the desired output format.
        # Providing a clear example is crucial for getting reliable, structured output.

        extraction_prompt = (
            f"You are a knowledge extraction engine. Your task is to extract Subject-Relation-Object triples from the given text. "
            f"Format your output strictly as a list of Python tuples. For example: [('Sam', 'works_for', 'Innovatech'), ('Innovatech', 'focuses_on', 'Energy')]. "
            f"If no triples are found, return an empty list [].\n\n"
            f"Text to analyze:\n{text}"
        )

        response = generate_text("You are an expert knowledge graph extractor.", extraction_prompt)
        # Safely parse the string representation of a list of tuples from the LLM's response.
        try:
            # Using regular expressions is a much safer alternative to `eval()`, as it avoids
            # executing arbitrary code that might be maliciously or accidentally included in the LLM's output.
            # This regex looks for patterns matching ('item1', 'item2', 'item3').
            found_triples = re.findall(r"\(['\"](.*?)['\"],\s*['\"](.*?)['\"],\s*['\"](.*?)['\"]\)", response)
            print(f"--- [Graph Memory: Extracted triples: {found_triples}] ---")
            return found_triples
        except Exception as e:
            # If parsing fails, log the error and return an empty list to prevent crashes.
            print(f"Could not parse triples from LLM response: {e}")
            return []

    def add_message(self, user_input: str, ai_response: str):
        full_text = f"User: {user_input}\nAI: {ai_response}"
        triples = self._extract_triples(full_text)

        for sub, rel, obj in triples:
            self.graph.add_edge(sub.strip(), obj.strip(), relation=rel.strip())

    def get_context(self, query: str) -> str:
        if not self.graph.nodes:
            return "The knowledge graph is empty."

        query_entities = [
            word.capitalize() for word in query.replace('?', '').split() if word.capitalize()
        ]

        if not query_entities:
            return "No relevant entities from your query were found in the knowledge graph."

        context_entities = []
        for entity in set(query_entities):
            for u, v, data in self.graph.out_edges(entity, data=True):
                context_entities.append(f"{u}-[{data['relation']}]-{v}")
            for u, v, data in self.graph.in_edges(entity, data=True):
                context_entities.append(f"{u}-[{data['relation']}]-{v}")

        return "### Facts Retrieved from Knowledge Graph:\n" + "\n".join(sorted(list(set(context_entities))))
