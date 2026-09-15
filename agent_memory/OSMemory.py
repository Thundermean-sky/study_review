from collections import deque

from main import BaseMemoryStrategy


# This conceptual strategy mimics how a computer's OS manages memory by using
# a small, fast 'active memory' (RAM) and a large, slower 'passive memory' (Disk).
# Information is "paged out" from active to passive memory when space is needed,
# and "paged in" when a query requires old information.
class OSMemory(BaseMemoryStrategy):
    def __init__(self, ram_size: int = 2):
        """
        Initializes the OS-like memory system.

        Args:
            ram_size: The maximum number of conversational turns to keep in active memory (RAM).
        """
        self.ram_size = ram_size
        # The 'RAM' is a deque, holding the most recent turns.
        self.active_memory = deque()
        # The 'Hard Disk' is a dictionary for storing paged-out turns.
        self.passive_memory = {}
        # A counter to give each turn a unique ID.
        self.turn_count = 0

    def add_message(self, user_input: str, ai_response: str):
        """Adds a turn to active memory, paging out the oldest turn to passive memory if RAM is full."""
        turn_id = self.turn_count
        turn_data = f"User: {user_input}\nAI: {ai_response}"

        # Check if active memory (RAM) is at capacity.
        if len(self.active_memory) >= self.ram_size:
            # If so, remove the least recently used (oldest) item from active memory.
            lru_turn_id, lru_turn_data = self.active_memory.popleft()
            # Move it to passive memory (the hard disk).
            self.passive_memory[lru_turn_id] = lru_turn_data
            print(f"--- [OS Memory: Paging out Turn {lru_turn_id} to passive storage.] ---")

        # Add the new turn to active memory.
        self.active_memory.append((turn_id, turn_data))
        self.turn_count += 1

    def get_context(self, query: str) -> str:
        """Provides RAM context and simulates a 'page fault' to pull from passive memory if needed."""
        # The base context is always what's in the active memory.
        active_context = "\n".join([data for _, data in self.active_memory])

        # Simulate a page fault: check if any words in the query match content in passive memory.
        # A real system would use embeddings for this, but keyword search demonstrates the concept.
        paged_in_context = ""
        for turn_id, data in self.passive_memory.items():
            if any(word in data.lower() for word in query.lower().split() if len(word) > 3):
                paged_in_context += f"\n(Paged in from Turn {turn_id}): {data}"
                print(f"--- [OS Memory: Page fault! Paging in Turn {turn_id} from passive storage.] ---")

        # Combine the active context with any paged-in context.
        return f"### Active Memory (RAM):\n{active_context}\n\n### Paged-In from Passive Memory (Disk):\n{paged_in_context}"

    def clear(self):
        """Clears both active and passive memory stores."""
        self.active_memory.clear()
        self.passive_memory = {}
        self.turn_count = 0
        print("OS-like memory cleared.")