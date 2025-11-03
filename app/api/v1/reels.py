from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from app.models.reel_in import ReelIn
from app.models.frame_analysis import FrameAnalysis
from app.services.reels_service import reels_service, ReelsService

router = APIRouter()

@router.post("/reels", response_model=FrameAnalysis, status_code=status.HTTP_200_OK)
async def analyze_reel(
    reel_in: ReelIn,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        analysis_result = await service.analyze_reel_video(reel_in)
        return analysis_result
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

@router.post("/reels/{request_id}/zip", status_code=status.HTTP_200_OK)
async def zip_reel_frames(
    request_id: str,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        zip_path = service.zip_frames(request_id)
        return {"message": "Frames zipped successfully.", "zip_path": str(zip_path)}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frames for the given request ID not found.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@router.get("/reels/{request_id}/download", response_class=FileResponse)
async def download_zipped_frames(
    request_id: str,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        zip_path = service.temp_dir / f"frames_{request_id}.zip"
        if not zip_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zipped frames for the given request ID not found. Please generate it first.")

        return FileResponse(path=zip_path, filename=f"frames_{request_id}.zip", media_type='application/zip')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
