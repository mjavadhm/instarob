import httpx
import yaml
import json
import os
import aiofiles
import asyncio
import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger
from app.models.reel_in import ReelIn
from app.models.frame_analysis import FrameAnalysis

logger = get_logger()

class ReelsService:
    def __init__(self):
        # Configure Gemini client
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Load LLM and prompt configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("product_search", {})

        prompt_file_path = prompt_dir / self.llm_config.get("prompt_file", "prompts/product_search.txt")
        with open(prompt_file_path, "r") as f:
            self.prompt = f.read()

        # Create a temporary directory for video downloads
        self.temp_dir = Path("/tmp/reels_videos")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a directory for storing frames
        self.frames_dir = Path("frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ReelsService initialized. Model: {self.llm_config.get('model')}")

    async def _download_video(self, url: str) -> Path:
        """Asynchronously downloads a video from a URL to a temporary local file."""
        file_name = os.path.basename(urlparse(url).path)
        # Sanitize file name to prevent directory traversal or other issues
        file_name = "".join(c for c in file_name if c.isalnum() or c in ('.', '_', '-'))
        local_path = self.temp_dir / file_name

        logger.info(f"Starting video download from {url} to {local_path}...")
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                    response.raise_for_status()
                    async with aiofiles.open(local_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)
                logger.info(f"Successfully downloaded video to {local_path}")
                return local_path
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while downloading video: {e.response.status_code} for URL {e.request.url}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Network error while downloading video: {e}")
                raise

    def _extract_and_save_frames(self, video_path: Path, analysis: FrameAnalysis, request_id: str):
        """Extracts frames from a video at given timestamps and saves them."""

        request_frame_dir = self.frames_dir / request_id
        request_frame_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return

        for frame_info in analysis.best_frames:
            timestamp_ms = frame_info.timestamp_seconds * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            ret, frame = cap.read()
            if ret:
                sanitized_product_name = "".join(c for c in analysis.product_category if c.isalnum() or c in ('_', '-')).rstrip()
                frame_filename = f"{sanitized_product_name}_{frame_info.rank}.jpg"
                frame_path = request_frame_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)
                logger.info(f"Saved frame at {frame_info.timestamp_seconds}s to {frame_path}")
            else:
                logger.warning(f"Could not read frame at {frame_info.timestamp_seconds}s")

        cap.release()

    async def analyze_reel_video(self, reel_in: ReelIn) -> FrameAnalysis:
        """
        Orchestrates the main workflow: download, analyze with Gemini, and parse.
        """
        video_path = None
        video_file = None
        try:
            # 1. Download the video
            video_path = await self._download_video(reel_in.reel_url)

            # 2. Upload video to Gemini and analyze (using asyncio.to_thread for blocking calls)
            logger.info(f"Uploading '{video_path.name}' to Gemini File API in a separate thread...")
            video_file = await asyncio.to_thread(
                genai.upload_file, path=video_path, display_name=video_path.name
            )
            logger.info(f"Completed upload for file '{video_file.display_name}' (URI: {video_file.uri})")

            model_name = self.llm_config.get("model", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)

            logger.info(f"Sending request to Gemini model '{model_name}'...")
            response = await model.generate_content_async([self.prompt, video_file])

            # 3. Parse the response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            logger.info("Parsing JSON response from Gemini.")
            analysis_data = json.loads(response_text)

            # 4. Validate with Pydantic model
            analysis_result = FrameAnalysis(**analysis_data)

            # 5. Extract and save frames
            await asyncio.to_thread(
                self._extract_and_save_frames, video_path, analysis_result, reel_in.request_id
            )

            return analysis_result

        except Exception as e:
            logger.error(f"An error occurred during video analysis: {e}", exc_info=True)
            raise RuntimeError("Failed to process and analyze the video.")

        finally:
            # 6. Clean up the downloaded file and the uploaded file on Gemini
            if video_path and os.path.exists(video_path):
                await asyncio.to_thread(os.remove, video_path)
                logger.info(f"Cleaned up temporary local file: {video_path}")
            if video_file:
                logger.info(f"Deleting uploaded file '{video_file.display_name}' from Gemini in a separate thread.")
                await asyncio.to_thread(genai.delete_file, video_file.name)

    def zip_frames(self, request_ids: List[str], zip_filename: str) -> Path:
        """Zips the frames for a given list of request IDs."""

        batch_dir = self.temp_dir / zip_filename
        batch_dir.mkdir(exist_ok=True)

        for request_id in request_ids:
            source_dir = self.frames_dir / request_id
            if source_dir.is_dir():
                # Copy the entire directory to the batch folder
                shutil.copytree(source_dir, batch_dir / request_id)
            else:
                logger.warning(f"Directory for request_id {request_id} not found, skipping.")

        zip_path_base = self.temp_dir / zip_filename
        zip_path = shutil.make_archive(str(zip_path_base), 'zip', str(batch_dir))

        # Clean up the temporary batch directory
        shutil.rmtree(batch_dir)

        return Path(zip_path)

reels_service = ReelsService()
