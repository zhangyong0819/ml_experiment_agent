import os
from langchain_deepseek import ChatDeepSeek


def create_deepseek_model() -> ChatDeepSeek:
    """Construct the official integration without contacting the API or logging credentials."""
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("DEEPSEEK_API_KEY is required for real mode")
    return ChatDeepSeek(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                        temperature=0, timeout=60, max_retries=0, max_tokens=2048)
