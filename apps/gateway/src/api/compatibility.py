from fastapi import APIRouter

from src.services.compatibility_manifest import compatibility_manifest


router = APIRouter(prefix="/compatibility", tags=["compatibility"])


@router.get("/responses")
async def responses_compatibility_manifest():
    return compatibility_manifest()
