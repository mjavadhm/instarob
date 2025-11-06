import httpx
import yaml
import json
import os
import aiofiles
import asyncio
import cv2
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger
from app.models.reel_in import ReelIn
from app.models.frame_analysis import FrameAnalysis
from app.services.product_service import product_service
from app.services.filter_service import filter_service
from app.services.cost_service import cost_service

logger = get_logger()

class ReelsService:
    def __init__(self):
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

        # Prepare generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.llm_config.get("temperature", 0.7),
            thinking_budget=self.llm_config.get("thinking_budget", 0)
        )

    async def _download_video(self, url: str, request_id: str) -> Path:
        """Asynchronously downloads a video from a URL to a temporary local file."""

        # Try to get a file extension from the URL path
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".mp4"  # Default to .mp4 if no extension is found

        file_name = f"{request_id}{ext}"
        local_path = self.temp_dir / file_name

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        logger.info(f"Starting video download from {url} to {local_path}...")
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as response:
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
                sanitized_product_name = "".join(c for c in analysis.identified_product if c.isalnum() or c in ('_', '-')).rstrip()
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
        request_start_time = time.time()
        total_cost = 0.0
        try:
            # 1. Download the video
            video_path = await self._download_video(str(reel_in.reel_url), reel_in.request_id)

            # 2. Upload video to Gemini and analyze (using asyncio.to_thread for blocking calls)
            logger.info(f"Uploading '{video_path.name}' to Gemini File API in a separate thread...")
            video_file = await asyncio.to_thread(
                genai.upload_file, path=video_path, display_name=video_path.name
            )
            logger.info(f"Completed upload for file '{video_file.display_name}' (URI: {video_file.uri})")

            # Wait for the file to be processed
            logger.info(f"Waiting for file '{video_file.display_name}' to be processed...")
            start_time = time.time()
            while video_file.state.name == "PROCESSING":
                if time.time() - start_time > 60: # 60-second timeout
                    raise TimeoutError("File processing timed out.")
                await asyncio.sleep(2)
                video_file = await asyncio.to_thread(genai.get_file, video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError("File processing failed on the server.")

            logger.info(f"File '{video_file.display_name}' is now in state: {video_file.state.name}")

            model_name = self.llm_config.get("model", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)

            # Create the full prompt including the caption
            full_prompt = f"{self.prompt}\n\nVideo Caption: {reel_in.caption}"

            # Log token usage for the prompt
            prompt_token_count_result = await model.count_tokens_async([full_prompt, video_file])
            prompt_token_count = prompt_token_count_result.total_tokens
            logger.info(f"Prompt token count: {prompt_token_count}")

            logger.info(f"Sending request to Gemini model '{model_name}'...")
            response = await model.generate_content_async(
                [full_prompt, video_file],
                generation_config=self.generation_config
            )

            # Log token usage for the response
            response_token_count_result = await model.count_tokens_async(response.text)
            response_token_count = response_token_count_result.total_tokens
            logger.info(f"Response token count: {response_token_count}")

            # Calculate cost for the first LLM call
            total_cost += cost_service.calculate_cost(
                model_name=model_name,
                prompt_tokens=prompt_token_count,
                response_tokens=response_token_count
            )

            # 3. Parse the response
            response_text = response.text.strip()
            logger.info(f"Raw LLM Response: {response_text}")
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            logger.info("Parsing JSON response from Gemini.")
            analysis_data = json.loads(response_text)

            # 4. Validate with Pydantic model
            analysis_result = FrameAnalysis(**analysis_data)
            analysis_result.prompt_token_count = prompt_token_count
            analysis_result.response_token_count = response_token_count


            # 5. Extract and save frames
            await asyncio.to_thread(
                self._extract_and_save_frames, video_path, analysis_result, reel_in.request_id
            )

            # 6. Search for products in all best frames concurrently
            search_tasks = []
            request_frame_dir = self.frames_dir / reel_in.request_id
            sanitized_product_name = "".join(c for c in analysis_result.identified_product if c.isalnum() or c in ('_', '-')).rstrip()

            for frame_info in analysis_result.best_frames:
                frame_filename = f"{sanitized_product_name}_{frame_info.rank}.jpg"
                frame_path = request_frame_dir / frame_filename

                if frame_path.exists():
                    logger.info(f"Queueing product search for {frame_path} with prompt '{analysis_result.identified_product}'")
                    task = product_service.search_product(
                        frame_path=frame_path,
                        text_prompt=analysis_result.identified_product
                    )
                    search_tasks.append(task)
                else:
                    logger.warning(f"Frame not found at {frame_path}, skipping product search for this frame.")

            if search_tasks:
                logger.info(f"Running {len(search_tasks)} product searches in parallel...")
                search_results = await asyncio.gather(*search_tasks)

                all_products = []
                all_searched_urls = []

                for result in search_results:
                    if result:
                        if result.get("products"):
                            all_products.extend(result["products"])
                        if result.get("searched_image_url"):
                            all_searched_urls.append(result["searched_image_url"])

                analysis_result.product_info = all_products
                analysis_result.searched_image_urls = all_searched_urls

                # 7. (New) Text search and filtering if a Persian query is available
                if analysis_result.search_query_persian:
                    logger.info(f"Persian search query found: '{analysis_result.search_query_persian}'. Starting text search.")

                    # Search Torob by text
                    text_search_results = await product_service.search_on_torob_by_text(analysis_result.search_query_persian)

                    # Combine and deduplicate image and text search results
                    combined_products = {}
                    
                    
                    for p in all_products:
                        p['source'] = 'image'
                        combined_products[p['random_key']] = p

                    
                    if text_search_results:
                        for p in text_search_results:
                            p['source'] = 'text'
                            if p['random_key'] not in combined_products:
                                combined_products[p['random_key']] = p

                    product_list_to_filter = list(combined_products.values())
                    logger.info(f"Combined list of {len(product_list_to_filter)} unique products (tagged by source) will be sent to LLM for filtering.")

                    
                    relevant_keys, filter_prompt_tokens, filter_response_tokens, filter_model_name = await filter_service.filter_products_with_llm(
                        products=product_list_to_filter,
                        search_query=analysis_result.search_query_persian,
                        identified_product=analysis_result.identified_product,
                        product_description=analysis_result.product_description
                    )

                    # Calculate cost for the filter LLM call
                    total_cost += cost_service.calculate_cost(
                        model_name=filter_model_name,
                        prompt_tokens=filter_prompt_tokens,
                        response_tokens=filter_response_tokens
                    )

                    # Aggregate token counts
                    analysis_result.prompt_token_count += filter_prompt_tokens
                    analysis_result.response_token_count += filter_response_tokens
                    logger.info(f"Aggregated token counts. Total Prompt: {analysis_result.prompt_token_count}, Total Response: {analysis_result.response_token_count}")

                    filtered_products = [p for p in product_list_to_filter if p['random_key'] in relevant_keys]
                    analysis_result.product_info = filtered_products
                    logger.info(f"Filtering complete. Final product count: {len(filtered_products)}")

                else:
                    logger.info("No Persian search query provided. Skipping text search and filtering.")
                    analysis_result.product_info = all_products 

            else:
                logger.warning("No product searches were queued.")
                analysis_result.product_info = []

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

            request_duration = time.time() - request_start_time
            logger.info(f"Request finished. Total duration: {request_duration:.2f}s, Total LLM cost: ${total_cost:.6f}")

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
