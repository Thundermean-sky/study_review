import abc
import os
from openai import OpenAI
import tiktoken
import time

API_KEY = "sk-cwgglepenvzmuricdobdbwnqksfbjplzkzstcuryqafxaxds"
BASE_URL = "https://api.siliconflow.cn/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

GENERATION_MODEL = "deepseek-ai/DeepSeek-V3.2"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the LLM API to generate a text response.
    """

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system", "content": system_prompt + ". Shortly answer.",
            },
            {
                "role": "user", "content": user_prompt,
            }
        ]
    )

    return response.choices[0].message.content


def generate_embedding(text: str) -> list[float]:
    """
    Generate a numerical embedding from the given text.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


tokenizer = tiktoken.get_encoding('cl100k_base')


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text.
    """

    return len(tokenizer.encode(text))


class BaseMemoryStrategy(abc.ABC):
    """
    Abstract base class for all memory strategies.
    """

    @abc.abstractmethod
    def add_message(self, user_input: str, ai_response: str):
        """
        An abstract method that must be implemented by subclasses.
        It's responsible for adding a new user-AI interaction to the memory store.
        """

        pass

    @abc.abstractmethod
    def get_context(self, query: str) -> str:
        """
        An abstract method that must be implemented by subclasses.
        It retrieves and formats the relevant context from memory to be sent to the LLM.
        The 'query' parameter allows some strategies (like retrieval) to fetch context
        that is specifically relevant to the user's latest input.
        """

        pass

    @abc.abstractmethod
    def clear(self):
        """
        An abstract method that must be implemented by subclasses.
        It provides a way to reset the memory, which is useful for starting new conversations.
        """
        pass


