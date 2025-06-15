import os
from langgraph.graph import StateGraph

from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import get_openrouter_base_url, get_openrouter_api_key

from pydantic import BaseModel
from typing import Annotated, Literal


class SerperSearchResults(BaseModel):
    """Search results from Google Serper."""

    link: Annotated[str, "URL of the search result"]
    snippet: Annotated[str, "Snippet of the search result"]


class WorkflowState(BaseModel):
    """State for the workflow."""

    image_url: Annotated[str, "URL of the image to be processed"]
    explanation: Annotated[
        str | None,
        "Explanation of the meme image, where the humor has occurred in the meme",
    ] = None
    genre: Annotated[
        str | None,
        "Genre of the meme image, e.g., 'political', 'entertainment, 'sports', 'other'",
    ] = None
    heroes: Annotated[list[str] | None, "Hero roles in the meme image"] = None
    villains: Annotated[list[str] | None, "Villain roles in the meme image"] = None
    victims: Annotated[list[str] | None, "Victim roles in the meme image"] = None
    other_roles: Annotated[
        list[str] | None, "Other roles in the meme image, if any"
    ] = None
    sentiment: Annotated[
        str | None,
        "Sentiment of the meme image, e.g., 'positive', 'negative', 'neutral'",
    ] = None


async def meme_overview(state: WorkflowState) -> Command[Literal["__end__", "translator_node"]]:
    """
    Overview of the meme image, including its explanation, genre, roles, and sentiment.
    """

    class ExpectedOutput(BaseModel):
        """Expected output for the meme overview."""

        explanation: Annotated[
            str,
            "A brief explanation of the meme image, where the humor has occurred in the meme",
        ]
        genre: Annotated[
            str,
            "Genre of the meme image, e.g., 'political', 'entertainment, 'sports', 'other'",
        ]
        heroes: Annotated[list[str], "Hero roles in the meme image"]
        villains: Annotated[list[str], "Villain roles in the meme image"]
        victims: Annotated[list[str], "Victim roles in the meme image"]
        other_roles: Annotated[list[str], "Other roles in the meme image"]
        sentiment: Annotated[
            str, "Sentiment of the meme image, e.g., 'positive', 'negative', 'neutral'"
        ]

    llm = ChatOpenAI(
        base_url=get_openrouter_base_url(),
        api_key=get_openrouter_api_key(),
        model="google/gemini-2.0-flash-001",
        temperature=0.5,
    ).with_structured_output(ExpectedOutput)

    with open("ai/prompts/meme_overview.md", "r") as f:
        system_prompt_text = f.read()
    system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Please extract all text from this image. If the text is in Bengali, preserve the Bengali characters. Return only the extracted text without any additional commentary.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    overview_chain = system_prompt | llm

    response = await overview_chain.ainvoke(input={"image_url": state.image_url})

    print("Response from meme overview chain:", response)

    if isinstance(response, ExpectedOutput):
        return Command(
            goto="translator_node",
            update={
                "explanation": response.explanation,
                "genre": response.genre,
                "heroes": response.heroes,
                "villains": response.villains,
                "victims": response.victims,
                "other_roles": response.other_roles,
                "sentiment": response.sentiment,
            },
        )
    else:
        return Command(goto="__end__")


async def translator(state: WorkflowState) -> Command[Literal["__end__"]]:
    """
    Translate the explanation of the meme image into Bengali.
    """

    class TranslationOutput(BaseModel):
        """Expected output for translation of the explanation."""

        translated_explanation: Annotated[str, "The explanation translated in Bengali"]

    translator_llm = ChatOpenAI(
        base_url=get_openrouter_base_url(),
        api_key=get_openrouter_api_key(),
        model="deepseek/deepseek-r1-0528:free",
    ).with_structured_output(TranslationOutput)

    translator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translation assistant that translates explanations into Bengali. "
                "Translate the provided explanation into Bengali, preserving the original meaning.",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Translate this explanation into Bengali:",
                    },
                    {"type": "text", "text": "{explanation}"},
                ],
            ),
        ]
    )

    translator_chain = translator_prompt | translator_llm
    translator_response = await translator_chain.ainvoke(
        {"explanation": state.explanation}
    )

    print("Translator response:", translator_response)

    if isinstance(translator_response, TranslationOutput):
        return Command(
            goto="__end__",
            update={"explanation": translator_response.translated_explanation},
        )
    else:
        return Command(goto="__end__")


workflow = StateGraph(WorkflowState)

workflow.add_node("meme_overview_node", meme_overview)
workflow.add_node("translator_node", translator)
workflow.set_entry_point("meme_overview_node")

AnnotatorAgent = workflow.compile()


print(AnnotatorAgent.get_graph().draw_ascii())
