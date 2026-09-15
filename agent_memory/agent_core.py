from main import *
from Sequential import SequentialMemory
from SlidingWindow import SlidingWindowMemory
from Summarization import SummaryMemory
from RetrievalMemory import RetrievalMemory
from MemoryAugmented import MemoryAugmented
from HierarchicalMemory import HierarchicalMemory


class AIAgent:
    def __init__(self, memory_strategy: BaseMemoryStrategy, system_prompt: str="You are a helpful AI assistant." ):
        self.memory_strategy = memory_strategy
        self.system_prompt = system_prompt

        print(f"Agent initialized with {type(memory_strategy).__name__}")

    def chat(self, user_input: str):
        """

        """

        print(f"\n{'='*25} NEW INTERACTION {'='*25}")
        print(f"User > {user_input}")

        start_time = time.time()
        context = self.memory_strategy.get_context(query=user_input)
        retrieval_time = time.time() - start_time

        full_user_prompt = f"### MEMORY CONTEXT:\n{context}\n\n### CURRENT REQUEST:\n{user_input}"

        prompt_tokens = count_tokens(self.system_prompt + full_user_prompt)
        print("\n--- Agent Debug Info ---")
        print(f"Memory Retrieval Time: {retrieval_time:.4f} seconds")
        print(f"Estimated Prompt Tokens: {prompt_tokens}")
        print(f"\n[Full Prompt Sent to LLM]:\n---\nSYSTEM: {self.system_prompt}\nUSER: {full_user_prompt}\n---")

        start_time = time.time()
        ai_response = generate_text(self.system_prompt, full_user_prompt)
        generate_time = time.time() - start_time

        self.memory_strategy.add_message(user_input, ai_response)

        print(f"\nAgent > {ai_response}")
        print(f"(LLM Generation Time: {generate_time:.4f} seconds)")
        print(f"{'='*70}")


if __name__ == '__main__':
    # seq = SequentialMemory()
    # slid_window = SlidingWindowMemory(window_size=2)
    # summary = SummaryMemory(summary_threshold=4)
    # retrieval = RetrievalMemory(k=2, embedding_dim=1024)
    # memory = MemoryAugmented(window_size=2)
    hierarchical_memory = HierarchicalMemory()
    agent = AIAgent(memory_strategy=hierarchical_memory)

    # --- Start the conversation ---
    # First turn: The user provides an important piece of information with a keyword ("remember").
    # This should trigger the promotion logic, saving this message to both short-term and long-term memory.
    agent.chat("Please remember my User ID is AX-7890.")
    # Second turn: A casual conversation topic. This is added to short-term memory.
    agent.chat("Let's chat about the weather. It's very sunny today.")
    # Third turn: Another casual topic. This pushes the first message (User ID) out of the
    # short-term sliding window memory.
    agent.chat("I'm planning to go for a walk later.")

    # --- Test the hierarchical retrieval ---
    # The User ID is now out of the working memory's window.
    # The agent must now rely on its long-term, retrieval-based memory.
    agent.chat("I need to log into my account, can you remind me of my ID?")
    # A successful agent will retrieve 'AX-7890' from its long-term memory because the initial
    # message was promoted due to the keyword "remember".