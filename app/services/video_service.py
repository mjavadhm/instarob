import httpx
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse
import os

from app.core.logging import get_logger

logger = get_logger()

class VideoService:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f)

        # Create a temporary directory for video downloads
        self.temp_dir = Path("/tmp/reels_videos")
        self.temp_dir.mkdir(parents=True, exist_ok=True)


    async def _download_video(self, url: str) -> Path:
        """Downloads a video from a URL to a temporary local file."""
        file_name = os.path.basename(urlparse(url).path)
        local_path = self.temp_dir / file_name

        logger.info(f"Downloading video from {url} to {local_path}...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Successfully downloaded video to {local_path}")
                return local_path
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while downloading video: {e}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Network error while downloading video: {e}")
                raise

    async def _analyze_video_with_gemini(self, video_path: Path) -> Dict[str, Any]:
        """
        Mocks the interaction with the Gemini API.
        In a real implementation, this would involve:
        1. Reading the prompt from the configured prompt_file.
        2. Initializing the Gemini client.
        3. Sending the video and the prompt to the model.
        4. Parsing the JSON response.
        """
        logger.info(f"Analyzing video '{video_path.name}' with model '{self.llm_config['product_search']['model']}'...")

        # Mocked response for MVP
        mock_response = {
            "file_name": video_path.name,
            "product_category": "Suction Cup Knife Sharpener",
            "best_frames": [
                {"rank": 1, "timestamp_seconds": 12.5, "description": "Ideal, stable, front-on shot on a clean background. Product is perfectly in focus and well-lit.", "bounding_box_percent": {"x_min": 42, "y_min": 70, "x_max": 58, "y_max": 88}},
                {"rank": 2, "timestamp_seconds": 34.5, "description": "Excellent close-up view showing the sharpening mechanism and 'Lucky Duck' front logo in high detail.", "bounding_box_percent": {"x_min": 34, "y_min": 68, "x_max": 67, "y_max": 100}},
                {"rank": 3, "timestamp_seconds": 3.0, "description": "Clear 3/4 top-down view, clearly showing the top suction-lock mechanism and product shape.", "bounding_box_percent": {"x_min": 28, "y_min": 42, "x_max": 69, "y_max": 85}},
                {"rank": 4, "timestamp_seconds": 16.5, "description": "Stable, front-on shot. Clear view of the product before the sharpening demonstration begins.", "bounding_box_percent": {"x_min": 42, "y_min": 70, "x_max": 58, "y_max": 88}},
                {"rank": 5, "timestamp_seconds": 58.0, "description": "Clear, stable view of the product in its use context, stable on the counter during the post-sharpening demo.", "bounding_box_percent": {"x_min": 42, "y_min": 70, "x_max": 58, "y_max": 88}}
            ]
        }

        logger.info("Successfully received analysis from mock Gemini API.")
        return mock_response

    async def process_video_url(self, url: str) -> Dict[str, Any]:
        """
        Orchestrates the video processing workflow.
        """
        try:
            video_path = await self._download_video(url)
            analysis_result = await self._analyze_video_with_gemini(video_path)

            # Clean up the downloaded file
            os.remove(video_path)
            logger.info(f"Cleaned up temporary file: {video_path}")

            return analysis_result
        except Exception as e:
            logger.error(f"An error occurred during video processing: {e}")
            # Re-raise or handle appropriately
            raise

video_service = VideoService()
