from langgraph.graph import StateGraph
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, Literal
from dotenv import load_dotenv
import os

load_dotenv()

from utils import get_openrouter_base_url, get_openrouter_api_key

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
search_tool = Tool(
    name="search",
    func=search.run,
    description="Useful for searching the web for information.",
)


class SerperSearchResults(BaseModel):
    """Search results from Google Serper."""

    link: Annotated[str, "URL of the search result"]
    snippet: Annotated[str, "Snippet of the search result"]


class WorkflowState(BaseModel):
    """State for the workflow."""

    image_url: Annotated[str, "URL of the image to be processed"]
    context: Annotated[
        str | None,
        "Extra information extracted from the web to better explain the meme image",
    ] = None


async def search_context(state: WorkflowState) -> Command[Literal["__end__"]]:
    """
    Search for context about the meme image using Google Serper.
    """

    class SearchKeywordOutput(BaseModel):
        """Expected output for search keyword generation."""

        search_keyword: Annotated[
            str, "The keyword to search for based on the meme image"
        ]

    # First, generate a search keyword based on the image
    keyword_llm = ChatOpenAI(
        base_url=get_openrouter_base_url(),
        api_key=get_openrouter_api_key(),
        model="google/gemini-2.0-flash-001",
        temperature=0.2,
    ).with_structured_output(SearchKeywordOutput)

    keyword_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant that analyzes meme images to identify the underlying real-world topic or event they reference."
                    "You do not describe the meme itself, nor the meme template, but rather generate search keywords that would help someone learn "
                    "about the actual subject or context the meme is referring to (e.g., historical events, political topics, cultural references)."
                ),
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "What is a good search keyword or phrase that describes the real-world topic this meme is referencing?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    keyword_chain = keyword_prompt | keyword_llm
    keyword_response = await keyword_chain.ainvoke({"image_url": state.image_url})

    if isinstance(keyword_response, SearchKeywordOutput):
        search_keyword = keyword_response.search_keyword
        search_result = search.run(search_keyword)

        print("---Search result:----", search_result)

        class BengaliTranslationOutput(BaseModel):
            """Expected output for translation of the search result."""

            translated_to_bengali: Annotated[
                str, "The search result translated in Bengali"
            ]

        # translator_llm = ChatOllama(
        #     model="qwen2.5vl:7b", temperature=0.1, base_url="http://localhost:11434"
        # ).with_structured_output(BengaliTranslationOutput)
        translator_llm = ChatOpenAI(
            base_url=get_openrouter_base_url(),
            api_key=get_openrouter_api_key(),
            model="google/gemini-2.0-flash-001",
        ).with_structured_output(BengaliTranslationOutput)
        translator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation assistant that translates text of snippets into Bengali. "
                "Translate the provided snippet into Bengali, preserving the original meaning.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Translate this snippet into Bengali:",
                    },
                    {"type": "text", "text": "{snippet}"},
                ],
            ),
        ]
    )
        print("Translator prompt:", translator_prompt)
        translator_chain = translator_prompt | translator_llm
        translator_response = await translator_chain.ainvoke({"snippet": search_result})

        print("Translator response:", translator_response)

        if isinstance(translator_response, BengaliTranslationOutput):
            return Command(
                goto="__end__",
                update={"context": translator_response.translated_to_bengali},
            )
        else:
            return Command(
                goto="__end__",
                update={"context": search_result},
            )

    return Command(goto="__end__")


workflow = StateGraph(WorkflowState)
workflow.add_node("search_context_node", search_context)
workflow.add_edge("__start__", "search_context_node")

ContextSearchAgent = workflow.compile()
