from fastapi import APIRouter, Depends, HTTPException, status
from app.models.reel_in import ReelIn
from app.models.suggestions_out import SuggestionsOut
from app.services.reels_service import reels_service, ReelsService
import asyncio

router = APIRouter()

@router.post("/", response_model=SuggestionsOut, status_code=status.HTTP_200_OK)
async def analyze_reel(
    reel_in: ReelIn,
    service: ReelsService = Depends(lambda: reels_service),
    timeout=60.0
):
    try:
        analysis_result = await asyncio.wait_for(
            service.analyze_reel_video(reel_in),
            timeout=60.0  # تایم‌اوت سراسری 60 ثانیه
        )
        return analysis_result
    except asyncio.TimeoutError:
        # logger.error("Request timed out after 60 seconds.") # خطای تایم‌اوت را لاگ کنید
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, # یا 408
            detail="Request processing timed out after 60 seconds."
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")
