import base64
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from supabase import AsyncClient
from db.config import get_supabase_client
from uuid import uuid4
import asyncio
import logging
from typing import List, Dict, Any
from pathlib import Path
from routes.annotation.annotation import router as annotation_router
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from utils import get_openrouter_base_url, get_openrouter_api_key

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(annotation_router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILES_PER_BATCH = 2500
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_CONCURRENT_UPLOADS = 50
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


async def extract_ocr_text_from_image(file_content: bytes, file_mime_type: str):
    base64_encoded_data = base64.b64encode(file_content).decode("utf-8")
    data_url = f"data:{file_mime_type};base64,{base64_encoded_data}"
    # llm = ChatOllama(
    #     model="qwen2.5vl:7b", temperature=0.1, base_url="http://localhost:11434"
    # )
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0.1,
    # )
    llm = ChatOpenAI(
        base_url=get_openrouter_base_url(),
        api_key=get_openrouter_api_key(),
        model="google/gemini-2.0-flash-001"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Please extract all text from this image. If the text is in Bengali, preserve the Bengali characters. Return only the extracted text without any additional commentary.",
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    llm_response = await llm.ainvoke([message])

    return llm_response.content


def validate_file(file: UploadFile) -> Dict[str, Any]:
    """Validate individual file before processing."""
    errors = []

    if not file.filename:
        errors.append("File name is required")

    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            errors.append(
                f"File extension '{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
            errors.append(f"MIME type '{file.content_type}' not allowed")

    if file.size and file.size > MAX_FILE_SIZE:
        errors.append(
            f"File size {file.size} exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )

    return {"valid": len(errors) == 0, "errors": errors}


async def check_file_status(supabase: AsyncClient, file_name: str) -> Dict[str, Any]:
    """Check if file exists in DB and storage, return status info."""
    try:
        db_response = (
            await supabase.table("annotated_memes")
            .select("image_id, file_name, annotation_status")
            .eq("file_name", file_name)
            .execute()
        )

        if not db_response.data:
            return {
                "exists_in_db": False,
                "exists_in_storage": False,
                "can_upload": True,
            }

        record = db_response.data[0]
        image_id = record["image_id"]

        try:
            storage_response = await supabase.storage.from_("memes").info(image_id)
            exists_in_storage = storage_response is not None
        except Exception:
            exists_in_storage = False

        return {
            "exists_in_db": True,
            "exists_in_storage": exists_in_storage,
            "can_upload": not exists_in_storage,
            "image_id": image_id,
            "current_status": record.get("annotation_status"),
        }

    except Exception as e:
        logger.error(f"Failed to check file status for '{file_name}': {e}")
        raise RuntimeError(f"Failed to check file status: {e}") from e


async def create_or_update_db_record(
    supabase: AsyncClient,
    file_name: str,
    file_content: bytes,
    file_mime_type: str,
    image_id: str | None = None,
) -> str:
    """Create new record or update existing record with 'uploading' status."""
    try:
        if not image_id:
            image_id = str(uuid4())

        # Try to update existing record first
        update_response = (
            await supabase.table("annotated_memes")
            .update(
                {
                    "annotation_status": "uploading",
                    "uploaded_meme_url": None,
                    "err_msg": None,
                }
            )
            .eq("image_id", image_id)
            .execute()
        )

        ocr_text = await extract_ocr_text_from_image(file_content, file_mime_type)

        # If no rows updated, create new record
        if not update_response.data:
            data = {
                "image_id": image_id,
                "file_name": file_name,
                "annotation_status": "uploading",
                "uploaded_meme_url": None,
                "err_msg": None,
                "ocr_text": ocr_text,
            }
            await supabase.table("annotated_memes").insert(data).execute()

        return image_id
    except Exception as e:
        logger.error(f"Failed to create/update DB record for '{file_name}': {e}")
        raise RuntimeError(f"Failed to create/update DB record: {e}") from e


async def update_status_success(supabase: AsyncClient, image_id: str, file_name: str):
    """Update database record on successful upload."""
    try:
        uploaded_url = (
            f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/memes/{image_id}"
        )
        await supabase.table("annotated_memes").update(
            {
                "annotation_status": "uploaded",
                "uploaded_meme_url": uploaded_url,
                "err_msg": None,
            }
        ).eq("image_id", image_id).execute()
    except Exception as e:
        logger.error(f"Failed to update success status for '{file_name}': {e}")
        # Don't raise here to avoid cascading failures


async def update_status_failed(
    supabase: AsyncClient, image_id: str, file_name: str, error_msg: str
):
    """Update database record on failed upload."""
    try:
        await supabase.table("annotated_memes").update(
            {"annotation_status": "upload_failed", "err_msg": error_msg}
        ).eq("image_id", image_id).execute()
    except Exception as e:
        logger.error(f"Failed to update error status for '{file_name}': {e}")
        # Don't raise here to avoid cascading failures


async def process_single_file(
    supabase: AsyncClient, file: UploadFile
) -> Dict[str, Any]:
    """Process a single file upload with complete error handling and resilient continuation."""
    file_name = file.filename
    image_id = None

    if not file_name:
        return {
            "filename": file_name,
            "status": "failed",
            "error": "File name is required",
            "action": "skipped",
        }

    try:
        # Validate file
        validation = validate_file(file)
        if not validation["valid"]:
            return {
                "filename": file_name,
                "status": "failed",
                "error": "; ".join(validation["errors"]),
                "action": "skipped",
            }

        file_status = await check_file_status(supabase, file_name)

        if file_status["exists_in_db"] and file_status["exists_in_storage"]:
            return {
                "filename": file_name,
                "status": "skipped",
                "message": "File already exists in database and storage",
                "action": "no_upload_needed",
            }
        elif file_status["exists_in_db"] and not file_status["exists_in_storage"]:
            image_id = file_status["image_id"]
            action = "upload_to_storage"
        else:
            image_id = None
            action = "new_upload"

        file_content = await file.read()
        if not file_content:
            raise ValueError("File content is empty")
        file_mime_type = file.content_type or "image/jpeg"

        image_id = await create_or_update_db_record(
            supabase, file_name, file_content, file_mime_type, image_id
        )

        file_options = {
            "content_type": file_mime_type,
            "cache_control": "3600",
        }

        await supabase.storage.from_("memes").upload(image_id, file_content, file_options)  # type: ignore

        await update_status_success(supabase, image_id, file_name)

        return {
            "filename": file_name,
            "status": "success",
            "image_id": image_id,
            "action": action,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to process file '{file_name}': {error_msg}")

        if image_id:
            try:
                await update_status_failed(supabase, image_id, file_name, error_msg)
            except Exception as update_error:
                logger.error(
                    f"Failed to update error status for '{file_name}': {update_error}"
                )

        return {
            "filename": file_name,
            "status": "failed",
            "error": error_msg,
            "action": "error_occurred",
        }


async def process_files_in_batches(
    supabase: AsyncClient, files: List[UploadFile]
) -> List[Dict[str, Any]]:
    """Process files in controlled batches to avoid overwhelming the system."""
    results = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    async def process_with_semaphore(file: UploadFile):
        async with semaphore:
            return await process_single_file(supabase, file)

    tasks = [process_with_semaphore(file) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(
                {
                    "filename": files[i].filename,
                    "status": "failed",
                    "error": f"Unexpected error: {str(result)}",
                }
            )
        else:
            processed_results.append(result)

    return processed_results


@app.post("/upload/memes")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple meme files with proper status tracking and error handling.
    Supports bulk uploads up to 2500 files with concurrent processing.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > MAX_FILES_PER_BATCH:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum allowed: {MAX_FILES_PER_BATCH}, received: {len(files)}",
            )

        supabase = await get_supabase_client()

        logger.info(f"Starting bulk upload of {len(files)} files")

        results = await process_files_in_batches(supabase, files)

        successful_uploads = [r for r in results if r["status"] == "success"]
        failed_uploads = [r for r in results if r["status"] == "failed"]
        skipped_uploads = [r for r in results if r["status"] == "skipped"]

        logger.info(
            f"Bulk upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed, {len(skipped_uploads)} skipped"
        )

        return {
            "total_files": len(files),
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "skipped_uploads": len(skipped_uploads),
            "results": results,
            "summary": {
                "success_rate": f"{(len(successful_uploads) / len(files)) * 100:.1f}%",
                "successful_files": [
                    {"filename": r["filename"], "action": r.get("action", "unknown")}
                    for r in successful_uploads
                ],
                "failed_files": [
                    {"filename": r["filename"], "error": r["error"]}
                    for r in failed_uploads
                ],
                "skipped_files": [
                    {
                        "filename": r["filename"],
                        "message": r.get("message", "Already exists"),
                    }
                    for r in skipped_uploads
                ],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk upload failed with unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        supabase = await get_supabase_client()
        # Simple query to test database connectivity
        await supabase.table("annotated_memes").select("count").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
