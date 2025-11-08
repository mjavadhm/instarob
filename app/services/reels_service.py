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
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
import base64
import google.generativeai as genai
import uuid


from app.core.config import settings
from app.core.logging import get_logger
from app.models.reel_in import ReelIn
from app.models.frame_analysis import FrameAnalysis
from app.models.caption_analysis import CaptionAnalysis
from app.models.suggestions_out import SuggestionsOut
from app.services.product_service import product_service
from app.services.filter_service import filter_service
from app.services.cost_service import cost_service
from app.services.openrouter_service import openrouter_service
from app.services.edenai_service import EdenAIService

logger = get_logger()

class ReelsService:
    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        prompt_dir = config_path.parent
        with open(config_path, "r") as f:
            self.llm_config = yaml.safe_load(f).get("product_search", {})

        prompt_file_path = prompt_dir.parent / "config" / self.llm_config.get("prompt_file", "prompts/product_search.txt")
        with open(prompt_file_path, "r") as f:
            self.video_prompt_template = f.read()

        self.temp_dir = Path("/tmp/reels_videos")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = Path("frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.edenai_service = EdenAIService()
        logger.info(f"ReelsService initialized. Model: {self.llm_config.get('model')}")

    async def _download_video(self, url: str, request_id: str) -> Path:
        local_path = self.temp_dir / f"{request_id}{os.path.splitext(urlparse(url).path)[1] or '.mp4'}"
        headers = {"User-Agent": "Mozilla/5.0"}
        logger.info(f"Starting video download from {url} to {local_path}...")
        try:
            async with httpx.AsyncClient() as client, client.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
            logger.info(f"Successfully downloaded video to {local_path}")
            return local_path
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading video: {e}", exc_info=True)
            raise

    def _extract_and_save_frames(self, video_path: Path, analysis: FrameAnalysis, request_id: str):
        if not analysis.identified_product:
            logger.info("No identified product, skipping frame extraction.")
            return

        request_frame_dir = self.frames_dir / request_id
        request_frame_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return
        sanitized_product_name = "".join(c for c in analysis.identified_product if c.isalnum() or c in ('_', '-')).rstrip()
        for frame_info in analysis.best_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_info.timestamp_seconds * 1000)
            ret, frame = cap.read()
            if ret:
                frame_path = request_frame_dir / f"{sanitized_product_name}_{frame_info.rank}.jpg"
                cv2.imwrite(str(frame_path), frame)
                logger.info(f"Saved frame at {frame_info.timestamp_seconds}s to {frame_path}")
        cap.release()

    def _get_frame_paths(self, request_id: str, analysis_result: FrameAnalysis) -> List[Path]:
        if not analysis_result.identified_product:
            return []
        request_frame_dir = self.frames_dir / request_id
        sanitized_product_name = "".join(c for c in analysis_result.identified_product if c.isalnum() or c in ('_', '-')).rstrip()
        return [
            request_frame_dir / f"{sanitized_product_name}_{frame_info.rank}.jpg"
            for frame_info in analysis_result.best_frames
            if (request_frame_dir / f"{sanitized_product_name}_{frame_info.rank}.jpg").exists()
        ]

    async def _analyze_video_path(self, reel_in: ReelIn, video_path: Path) -> Tuple[Optional[FrameAnalysis], int, int, float]:
        try:
            async with aiofiles.open(video_path, "rb") as f:
                video_bytes = await f.read()
            video_base64 = base64.b64encode(video_bytes).decode("utf-8")

            full_prompt = f"{self.video_prompt_template}\n\nVideo Caption: {reel_in.text}"

            edenai_response = await self.edenai_service.analyze_video_frames(video_base64, full_prompt)

            response_content = edenai_response['choices'][0]['message']['content']
            logger.info(f"Raw EdenAI Response: {response_content}")
            response_text = response_content.strip().removeprefix("```json").removesuffix("```")

            analysis_data = json.loads(response_text)
            analysis_result = FrameAnalysis(**analysis_data)

            prompt_tokens = edenai_response['usage']['prompt_tokens']
            response_tokens = edenai_response['usage']['completion_tokens']
            cost = edenai_response.get('cost', 0.0)
            # logger.info(f"Raw EdenAI Response: {response_content}")
            logger.info(f"EdenAI Response Cost: {cost}")

            return analysis_result, prompt_tokens, response_tokens, cost
        except Exception as e:
            logger.error(f"Error in _analyze_video_path with EdenAI: {e}", exc_info=True)
            return None, 0, 0, 0.0

    async def _analyze_caption_path(self, reel_in: ReelIn) -> Tuple[Optional[CaptionAnalysis], int, int]:
        if not reel_in.text:
            return None, 0, 0
        return await openrouter_service.analyze_caption_for_search_query(reel_in.text)

    async def analyze_reel_video(self, reel_in: ReelIn) -> SuggestionsOut:
        request_id = str(uuid.uuid4())
        request_start_time = time.time()
        video_path = None
        total_cost = 0.0
        prompt_token_total = 0
        response_token_total = 0

        try:
            video_path = await self._download_video(str(reel_in.url), request_id)

            # --- Parallel Analysis ---
            video_analysis_task = asyncio.create_task(self._analyze_video_path(reel_in, video_path))
            caption_analysis_task = asyncio.create_task(self._analyze_caption_path(reel_in))

            results = await asyncio.gather(video_analysis_task, caption_analysis_task, return_exceptions=True)

            # --- Process Video Analysis Results ---
            video_result = results[0]
            analysis_result = None
            frame_paths = []

            if isinstance(video_result, Exception):
                logger.error(f"Video analysis path failed critically: {video_result}", exc_info=video_result)
            elif not video_result[0]:
                logger.warning("Video analysis returned no result.")
            else:
                analysis_result, v_prompt, v_resp, v_cost = video_result
                prompt_token_total += v_prompt
                response_token_total += v_resp
                total_cost += v_cost
                await asyncio.to_thread(self._extract_and_save_frames, video_path, analysis_result, request_id)
                frame_paths = await asyncio.to_thread(self._get_frame_paths, request_id, analysis_result)
            
            if analysis_result is None:
                analysis_result = FrameAnalysis(identified_product=None, best_frames=[])

            # --- Process Caption Analysis Results ---
            caption_result = results[1]
            caption_search_query = None
            if isinstance(caption_result, Exception):
                logger.warning(f"Caption analysis path failed: {caption_result}", exc_info=caption_result)
            else:
                caption_analysis, c_prompt, c_resp = caption_result
                prompt_token_total += c_prompt
                response_token_total += c_resp
                if caption_analysis and caption_analysis.search_query_persian:
                    caption_search_query = caption_analysis.search_query_persian
                    total_cost += cost_service.calculate_cost(openrouter_service.caption_analysis_model, c_prompt, c_resp)

            # --- Concurrent Searches ---
            image_search_tasks = [product_service.search_product(fp, analysis_result.identified_product) for fp in frame_paths]

            text_search_task = None
            if caption_search_query:
                text_search_task = product_service.search_on_torob_by_text(caption_search_query)

            search_results = await asyncio.gather(*image_search_tasks, text_search_task, return_exceptions=True)

            # --- Process Search Results ---
            image_search_results = [r for r in search_results[:len(image_search_tasks)] if not isinstance(r, Exception)]

            text_search_products = []
            if text_search_task:
                text_result = search_results[-1]
                if isinstance(text_result, Exception):
                    logger.warning(f"Text search failed: {text_result}", exc_info=text_result)
                elif text_result:
                    text_search_products = text_result

            # --- Filter Text Search (if applicable) ---
            filtered_text_products = []
            if text_search_products:
                keys, or_p, or_c = await openrouter_service.filter_text_search_results(text_search_products, frame_paths, analysis_result.identified_product)
                prompt_token_total += or_p
                response_token_total += or_c
                total_cost += cost_service.calculate_cost(openrouter_service.text_filter_model, or_p, or_c)

                keys_map = {key: i for i, key in enumerate(keys)}
                filtered_text_products = sorted([p for p in text_search_products if p['random_key'] in keys_map], key=lambda p: keys_map[p['random_key']])[:5]

            # --- Combine, Deduplicate, and Final Rank ---
            candidate_products = {p['random_key']: {**p, 'source': 'text'} for p in filtered_text_products}

            all_urls = []
            for result in image_search_results:
                if result and result.get("products"):
                    url = result.get("searched_image_url")
                    if url: all_urls.append(url)
                    for p in result["products"][:5]:
                        if p['random_key'] not in candidate_products:
                            candidate_products[p['random_key']] = {**p, 'source': 'image', 'image_url': url}

            product_list_to_rank = list(candidate_products.values())
            final_keys = []
            if not product_list_to_rank:
                logger.info("No candidate products found to rank. Skipping final ranking.")
            else:
                ranked_keys, r_p, r_c, r_model = await filter_service.filter_and_rank_products(
                    products=product_list_to_rank,
                    ground_truth_frame_paths=frame_paths,
                    identified_product=analysis_result.identified_product,
                    product_description=analysis_result.product_description
                )
                prompt_token_total += r_p
                response_token_total += r_c
                total_cost += cost_service.calculate_cost(r_model, r_p, r_c)
                final_keys = ranked_keys

            return SuggestionsOut(suggestions=final_keys)

        except Exception as e:
            logger.error(f"Critical error in video analysis workflow: {e}", exc_info=True)
            return SuggestionsOut(suggestions=[])
        finally:
            request_frame_dir = self.frames_dir / request_id
            if request_frame_dir.exists():
                await asyncio.to_thread(shutil.rmtree, request_frame_dir)
            if video_path and os.path.exists(video_path):
                await asyncio.to_thread(os.remove, video_path)
            duration = time.time() - request_start_time
            logger.info(f"Request finished. Duration: {duration:.2f}s, Total Cost: ${total_cost:.6f}")
            logger.info(f"Final token counts. Prompt: {prompt_token_total}, Response: {response_token_total}")

    def zip_frames(self, request_ids: List[str], zip_filename: str) -> Path:
        batch_dir = self.temp_dir / zip_filename
        batch_dir.mkdir(exist_ok=True)
        for request_id in request_ids:
            source_dir = self.frames_dir / request_id
            if source_dir.is_dir():
                shutil.copytree(source_dir, batch_dir / request_id)
        zip_path = shutil.make_archive(str(self.temp_dir / zip_filename), 'zip', str(batch_dir))
        shutil.rmtree(batch_dir)
        return Path(zip_path)

reels_service = ReelsService()
