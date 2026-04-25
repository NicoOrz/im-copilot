import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("VOLC_MODEL", "ep-20260422180225-zllc4"),
        api_key=os.getenv("VOLC_API_KEY"),
        base_url=os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        **kwargs,
    )
