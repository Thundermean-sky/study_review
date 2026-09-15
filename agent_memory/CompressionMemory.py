from main import BaseMemoryStrategy, generate_text


# --- Strategy 8: Compression & Consolidation Memory ---
# This strategy aggressively reduces token usage by using the LLM to compress
# each conversational turn into a single, dense, factual statement.
# This is a more extreme version of summarization, focused purely on
# information density over conversational flow.
class CompressionMemory(BaseMemoryStrategy):
    def __init__(self):
        """Initializes the memory with an empty list to store compressed facts."""
        self.compressed_facts = []

    def add_message(self, user_input: str, ai_response: str):
        """Uses the LLM to compress the latest turn into a concise factual statement."""
        # Combine the user and AI messages into a single text block for compression.
        text_to_compress = f"User: {user_input}\nAI: {ai_response}"

        # This prompt is highly specific, instructing the LLM to act as a "data compressor"
        # and to be as concise as possible.
        compression_prompt = (
            f"You are a data compression engine. Your task is to distill the following text into its most essential, factual statement. "
            f"Be as concise as possible, removing all conversational fluff. For example, 'User asked about my name and I, the AI, responded that my name is an AI assistant' should become 'User asked for AI's name.'\n\n"
            f"Text to compress:\n\"{text_to_compress}\""
        )

        # Call the LLM with the compression persona.
        compressed_fact = generate_text("You are an expert data compressor.", compression_prompt)
        print(f"--- [Compression Memory: New fact stored: '{compressed_fact}'] ---")
        # Add the highly compressed fact to our memory list.
        self.compressed_facts.append(compressed_fact)

    def get_context(self, query: str) -> str:
        """Returns the list of all compressed facts, formatted as a bulleted list."""
        if not self.compressed_facts:
            return "No compressed facts in memory."

        # The context is a simple, clean list of the core facts from the conversation.
        return "### Compressed Factual Memory:\n- " + "\n- ".join(self.compressed_facts)
