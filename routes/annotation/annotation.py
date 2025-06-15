from fastapi.routing import APIRouter
from pydantic import BaseModel
from ai.annotator_agent import AnnotatorAgent
from ai.context_search_agent import ContextSearchAgent
from db.config import get_supabase_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/annotation",
    tags=["annotation"],
)


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the annotation service.
    """
    return {"status": "ok", "message": "Annotation service is running."}


class RequestModel(BaseModel):
    meme_id: str
    meme_url: str


@router.post("/annotate")
async def annotate_meme(request: RequestModel):
    print(f"Received request to annotate meme: {request}")
    try:
        response = await AnnotatorAgent.ainvoke(input={"image_url": request.meme_url})
    except Exception as e:
        return {"error": str(e), "message": "Agent failed to process the meme."}

    explanation = response["explanation"]
    genre = response["genre"]
    heroes = response["heroes"]
    villains = response["villains"]
    victims = response["victims"]
    other_roles = response["other_roles"]
    sentiment = response["sentiment"]

    data = {
        "explanation": explanation,
        "genre": genre,
        "heroes": heroes,
        "villains": villains,
        "victims": victims,
        "other_roles": other_roles,
        "sentiment": sentiment,
        "annotation_status": "half_annotated",
    }
    try:
        supabase = await get_supabase_client()
        await supabase.table("annotated_memes").update(data).eq(
            "id", request.meme_id
        ).execute()
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to update meme annotation in the database.",
        }
    return response


@router.post("/generate-context")
async def extract_context(request: RequestModel):
    """
    Extract context for a meme image using the annotation agent.
    """
    print(f"Received request to extract context for meme: {request}")
    try:
        response = await ContextSearchAgent.ainvoke(
            input={"image_url": request.meme_url}
        )
    except Exception as e:
        return {"error": str(e), "message": "Agent failed to process the meme."}

    context = response["context"]
    if not context:
        return {"message": "No context found for the meme."}

    data = {
        "context": context,
        "annotation_status": "fully_annotated",
    }
    try:
        supabase = await get_supabase_client()
        await supabase.table("annotated_memes").update(data).eq(
            "id", request.meme_id
        ).execute()
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to update meme context in the database.",
        }

    return {"context": context}
