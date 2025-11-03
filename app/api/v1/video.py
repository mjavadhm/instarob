from fastapi import APIRouter, Depends, HTTPException, Body
from app.services.video_service import video_service, VideoService
from typing import Dict, Any
from pydantic import BaseModel, HttpUrl

router = APIRouter()

class VideoRequest(BaseModel):
    video_url: HttpUrl

@router.post("/video/process", response_model=Dict[str, Any], status_code=200)
async def process_video(
    request: VideoRequest,
    service: VideoService = Depends(lambda: video_service)
):
    try:
        result = await service.process_video_url(str(request.video_url))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
