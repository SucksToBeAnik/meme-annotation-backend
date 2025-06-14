from supabase import create_async_client
from dotenv import load_dotenv
load_dotenv()
import os


async def get_supabase_client():
    try:
        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not anon_key:
            raise ValueError(f"{"Anon Key" if not anon_key else "Supabase URL"} is not set in environment variables.")

        client = await create_async_client(url, anon_key)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create Supabase client: {e}") from e