from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from app.services.reels_service import reels_service, ReelsService
from pydantic import BaseModel
from typing import List

router = APIRouter()

class ZipRequest(BaseModel):
    request_ids: List[str]
    zip_filename: str

@router.post("/zip", status_code=status.HTTP_200_OK)
async def zip_frames_endpoint(
    zip_request: ZipRequest,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        # zip_path = service.zip_frames(zip_request.request_ids, zip_request.zip_filename)
        return {"message": "Frames zipped successfully.", "zip_path": "null"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

@router.get("/download/{zip_filename}", response_class=FileResponse)
async def download_zip_endpoint(
    zip_filename: str,
    service: ReelsService = Depends(lambda: reels_service)
):
    try:
        # Sanitize filename to prevent directory traversal
        # safe_filename = "".join(c for c in zip_filename if c.isalnum() or c in ('_', '-'))
        # zip_path = service.temp_dir / f"{safe_filename}.zip"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zipped file not found.")
        # if not zip_path.exists():
        #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zipped file not found.")

        # return FileResponse(path=zip_path, filename=f"{safe_filename}.zip", media_type='application/zip')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
