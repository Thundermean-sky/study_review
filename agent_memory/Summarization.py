from main import BaseMemoryStrategy
from main import generate_text

class SummaryMemory(BaseMemoryStrategy):
    def __init__(self, summary_threshold: int = 4):
        """
        Initializes the summarization memory.

        Args:
            summary_threshold: The number of messages (user + AI) to accumulate in the
                             buffer before triggering a summarization.
        """
        "触发总结的对话条数"
        self.summary_threshold = summary_threshold
        "临时记录存储空间"
        self.buffer = []
        "存储迄今为止的持续性对话总结"
        self.running_summary = ""

    def add_message(self, user_input: str, ai_response: str):
        self.buffer.append({
            "role": "user",
            "content": user_input,
        })

        self.buffer.append({
            "role": "assistant",
            "content": ai_response,
        })

        if len(self.buffer) >= self.summary_threshold:
            self._consolidate_summary()


    def _consolidate_summary(self):
        buffer_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.buffer])

        summarization_prompt = (
            f"You are a summarization expert. Your task is to create a concise summary of a conversation. "
            f"Combine the 'Previous Summary' with the 'New Conversation' into a single, updated summary. "
            f"Capture all key facts, names, and decisions.\n\n"
            f"### Previous Summary:\n{self.running_summary}\n\n"
            f"### New Conversation:\n{buffer_text}\n\n"
            f"### Updated Summary:"
        )

        new_summary = generate_text("You are an expert summarization engine.", summarization_prompt)
        self.running_summary = new_summary

        self.buffer.clear()

        print("New summary is: ", new_summary)

    def get_context(self, query: str) -> str:
        """
        Constructs the context to be sent to the LLM. It combines the long-term
        running summary with the short-term buffer of recent messages.
        The 'query' parameter is ignored as this strategy provides a general context.
        """
        # Format the messages currently in the buffer.
        buffer_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.buffer])
        # Return a combined context of the historical summary and the most recent, not-yet-summarized messages.
        return f"### Summary of Past Conversation:\n{self.running_summary}\n\n### Recent Messages:\n{buffer_text}"

    def clear(self):
        self.buffer.clear()


























