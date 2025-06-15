from langgraph.graph import StateGraph

from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import Chat

from pydantic import BaseModel
from typing import Annotated, Literal


class SerperSearchResults(BaseModel):
    """Search results from Google Serper."""

    link: Annotated[str, "URL of the search result"]
    snippet: Annotated[str, "Snippet of the search result"]


class WorkflowState(BaseModel):
    """State for the workflow."""

    image_url: Annotated[str, "URL of the image to be processed"]
    search_keyword: Annotated[
        str | None, "Keyword used for searching the meme image context"
    ] = None
    search_results: Annotated[
        list[SerperSearchResults] | None, "Search results from Google Serper"
    ] = None
    context: Annotated[
        str | None,
        "Extra information extracted from the web to better explain the meme image",
    ] = None
    final_context_url: Annotated[
        str | None, "URL of the final context, if available"
    ] = None
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


async def meme_overview(state: WorkflowState) -> Command[Literal["__end__"]]:
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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
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
            goto="__end__",
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


workflow = StateGraph(WorkflowState)

workflow.add_node("meme_overview_node", meme_overview)
workflow.set_entry_point("meme_overview_node")

AnnotatorAgent = workflow.compile()


print(AnnotatorAgent.get_graph().draw_ascii())
