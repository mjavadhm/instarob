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
from app.services.openrouter_service import openrouter_service
from app.core.utils import async_retry

logger = get_logger()

class ReelsService:
    def __init__(self):
        # Load LLM and prompt configuration
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("product_search", {})

        prompt_file_path = prompt_dir.parent / "config" / self.llm_config.get("prompt_file", "prompts/product_search.txt")
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
            temperature=self.llm_config.get("temperature", 0.7)
        )

    @async_retry()
    async def _download_video(self, url: str, request_id: str) -> Path:
        """Asynchronously downloads a video from a URL to a temporary local file."""
        path = urlparse(url).path
        ext = os.path.splitext(path)[1] or ".mp4"
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

    def _get_frame_paths(self, request_id: str, analysis_result: FrameAnalysis) -> List[Path]:
        """Returns a list of paths to the saved frames for a given request."""
        frame_paths = []
        request_frame_dir = self.frames_dir / request_id
        sanitized_product_name = "".join(c for c in analysis_result.identified_product if c.isalnum() or c in ('_', '-')).rstrip()
        for frame_info in analysis_result.best_frames:
            frame_filename = f"{sanitized_product_name}_{frame_info.rank}.jpg"
            frame_path = request_frame_dir / frame_filename
            if frame_path.exists():
                frame_paths.append(frame_path)
        return frame_paths

    async def _task_a_analyze_caption_and_search_text(self, reel_in: ReelIn):
        """Task A: Analyzes caption and performs text search."""
        try:
            query, p_tokens, r_tokens = await openrouter_service.analyze_caption(reel_in.caption)
            cost = cost_service.calculate_cost(openrouter_service.caption_analysis_model, p_tokens, r_tokens)
            if query:
                results = await product_service.search_on_torob_by_text(query)
                return results, p_tokens, r_tokens, cost
            return None, p_tokens, r_tokens, cost
        except Exception as e:
            logger.error(f"Error in text pipeline (Task A): {e}", exc_info=True)
            return None, 0, 0, 0.0

    async def _task_b_process_video_and_search_images(self, reel_in: ReelIn):
        """Task B: Downloads, analyzes video, extracts frames, and starts image searches."""
        video_path, video_file = None, None
        try:
            video_path = await self._download_video(str(reel_in.reel_url), reel_in.request_id)
            logger.info(f"Uploading '{video_path.name}' to Gemini File API...")
            video_file = await asyncio.to_thread(genai.upload_file, path=video_path, display_name=video_path.name)

            logger.info(f"Waiting for file '{video_file.display_name}' to be processed...")
            start_time = time.time()
            while video_file.state.name == "PROCESSING":
                if time.time() - start_time > 120: raise TimeoutError("File processing timed out.")
                await asyncio.sleep(5)
                video_file = await asyncio.to_thread(genai.get_file, video_file.name)
            if video_file.state.name == "FAILED": raise RuntimeError("File processing failed on the server.")

            model_name = self.llm_config.get("model", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            full_prompt = f"{self.prompt}\n\nVideo Caption: {reel_in.caption}"

            response = await model.generate_content_async([full_prompt, video_file], generation_config=self.generation_config)

            response_text = response.text.strip().removeprefix("```json").removesuffix("```")
            logger.info(f"Raw Gemini Response: {response_text}")
            analysis_data = json.loads(response_text)
            analysis_result = FrameAnalysis(**analysis_data)

            p_tokens = (await model.count_tokens_async([full_prompt, video_file])).total_tokens
            r_tokens = (await model.count_tokens_async(response.text)).total_tokens
            cost = cost_service.calculate_cost(model_name, p_tokens, r_tokens)
            analysis_result.prompt_token_count = p_tokens
            analysis_result.response_token_count = r_tokens

            await asyncio.to_thread(self._extract_and_save_frames, video_path, analysis_result, reel_in.request_id)
            frame_paths = self._get_frame_paths(reel_in.request_id, analysis_result)

            search_tasks = [product_service.search_product(fp, analysis_result.identified_product) for fp in frame_paths]
            image_search_task = asyncio.gather(*search_tasks)

            return analysis_result, frame_paths, image_search_task, cost, video_path, video_file
        except Exception as e:
            logger.error(f"Error in video pipeline (Task B): {e}", exc_info=True)
            # Cleanup resources if they were created before the error
            if video_path and os.path.exists(video_path):
                await asyncio.to_thread(os.remove, video_path)
            if video_file:
                await asyncio.to_thread(genai.delete_file, video_file.name)
            raise # Re-raise to let the orchestrator know this path failed

    async def analyze_reel_video(self, reel_in: ReelIn) -> FrameAnalysis:
        """Orchestrates the new parallel processing pipeline."""
        request_start_time = time.time()
        video_path, video_file = None, None
        final_analysis = FrameAnalysis() # Create a default result object

        try:
            # 1. Start Text and Video pipelines concurrently
            task_a = self._task_a_analyze_caption_and_search_text(reel_in)
            task_b = self._task_b_process_video_and_search_images(reel_in)

            results = await asyncio.gather(task_a, task_b, return_exceptions=True)

            # Unpack results and handle potential errors
            text_result, video_result = results

            if isinstance(text_result, Exception):
                logger.error("Text analysis task failed.", exc_info=text_result)
                text_search_results, txt_p, txt_r, txt_c = [], 0, 0, 0.0
            else:
                text_search_results, txt_p, txt_r, txt_c = text_result

            if isinstance(video_result, Exception):
                logger.error("Video processing task failed.", exc_info=video_result)
                raise RuntimeError("Core video processing pipeline failed, cannot continue.")
            else:
                final_analysis, frame_paths, image_search_task, vid_c, video_path, video_file = video_result

            final_analysis.prompt_token_count += txt_p
            final_analysis.response_token_count += txt_r
            total_cost = txt_c + vid_c

            # 2. Synchronization Point 1: Filter text results
            filtered_text_products = []
            if text_search_results and frame_paths:
                keys, p, r = await openrouter_service.filter_text_search_results(
                    text_search_results, frame_paths, final_analysis.identified_product
                )
                keys_map = {key: i for i, key in enumerate(keys)}
                sorted_prods = sorted([p for p in text_search_results if p['random_key'] in keys_map], key=lambda p: keys_map[p['random_key']])
                filtered_text_products = sorted_prods[:5]

                final_analysis.prompt_token_count += p
                final_analysis.response_token_count += r
                total_cost += cost_service.calculate_cost(openrouter_service.text_filter_model, p, r)

            # 3. Synchronization Point 2: Wait for image searches to complete
            image_search_results = await image_search_task

            # 4. Combine, Deduplicate, and Final Rank
            candidate_products = {p['random_key']: {**p, 'source': 'text'} for p in filtered_text_products}

            urls = []
            for result in image_search_results:
                if result and result.get("products"):
                    url = result.get("searched_image_url")
                    if url: urls.append(url)
                    for p in result["products"][:5]:
                        if p['random_key'] not in candidate_products:
                            candidate_products[p['random_key']] = {**p, 'source': 'image', 'image_url': url}
            final_analysis.searched_image_urls = list(set(urls))

            product_list = list(candidate_products.values())
            if product_list:
                keys, p, r, model = await filter_service.rank_products_with_llm(
                    product_list, frame_paths, final_analysis.identified_product
                )
                final_analysis.prompt_token_count += p
                final_analysis.response_token_count += r
                total_cost += cost_service.calculate_cost(model, p, r)

                prod_map = {p['random_key']: p for p in product_list}
                final_analysis.product_info = [prod_map[key] for key in keys if key in prod_map]

            return final_analysis

        except Exception as e:
            logger.error(f"An error occurred during the main analysis workflow: {e}", exc_info=True)
            # Ensure final_analysis is returned even on failure
            return final_analysis
        finally:
            if video_path and os.path.exists(video_path):
                await asyncio.to_thread(os.remove, video_path)
            if video_file:
                await asyncio.to_thread(genai.delete_file, video_file.name)

            duration = time.time() - request_start_time
            logger.info(f"Request finished. Duration: {duration:.2f}s, Total Cost: ${total_cost:.6f}")
            logger.info(f"Final token counts. Prompt: {final_analysis.prompt_token_count}, Response: {final_analysis.response_token_count}")

    def zip_frames(self, request_ids: List[str], zip_filename: str) -> Path:
        """Zips the frames for a given list of request IDs."""
        batch_dir = self.temp_dir / zip_filename
        batch_dir.mkdir(exist_ok=True)
        for request_id in request_ids:
            source_dir = self.frames_dir / request_id
            if source_dir.is_dir():
                shutil.copytree(source_dir, batch_dir / request_id)
            else:
                logger.warning(f"Directory for request_id {request_id} not found, skipping.")
        zip_path = shutil.make_archive(str(self.temp_dir / zip_filename), 'zip', str(batch_dir))
        shutil.rmtree(batch_dir)
        return Path(zip_path)

reels_service = ReelsService()
