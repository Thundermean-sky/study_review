from main import BaseMemoryStrategy
from SlidingWindow import SlidingWindowMemory
from main import generate_text


class MemoryAugmented(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2):
        """
        Initializes the memory-augmented system.

        Args:
            window_size: The number of recent turns to keep in the short-term memory.
        """
        self.sliding_window = SlidingWindowMemory(window_size=window_size)
        self.memory_token = []

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds the latest turn to recent memory and then uses an LLM call to decide
        if a new, persistent memory token should be created from this interaction.
        """
        self.sliding_window.add_message(user_input, ai_response)

        # Construct a prompt for the LLM to analyze the conversation turn and
        # determine if it contains a core fact worth remembering long-term.
        fact_extraction_prompt = (
            f"Analyze the following conversation turn. Does it contain a core fact, preference, or decision that should be remembered long-term? "
            f"Examples include user preferences ('I hate flying'), key decisions ('The budget is $1000'), or important facts ('My user ID is 12345').\n\n"
            f"Conversation Turn:\nUser: {user_input}\nAI: {ai_response}\n\n"
            f"If it contains such a fact, state the fact concisely in one sentence. Otherwise, respond with 'No important fact.'"
        )

        extract_fact = generate_text("You are a fact-extraction expert", fact_extraction_prompt)
        if "no important fact." not in extract_fact.lower():
            # If a fact was found, print a debug message and add it to our list of memory tokens.
            print(f"--- [Memory Augmentation: New memory token created: '{extract_fact}'] ---")
            self.memory_token.append(extract_fact)

    def get_context(self, query: str) -> str:
        recent_text = self.sliding_window.get_context(query)
        memory_token_context = "\n".join([f"- {token}" for token in self.memory_token])

        # Return the combined context, clearly separating the long-term facts from the recent chat.
        return f"### Key Memory Tokens (Long-Term Facts):\n{memory_token_context}\n\n### Recent Conversation:\n{recent_text}"

    def clear(self):
        self.memory_token = []
        self.sliding_window.clear()
