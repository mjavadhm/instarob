import aiofiles
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
import asyncio

from app.core.logging import get_logger

logger = get_logger()

class ProductService:
    def __init__(self):
        self.torob_headers = {
            "x-hackathon-key": "KDwavVLtDr4qwEiohTsYBcNW3MxDxbBxGfdaoGHF8FDuqBcD3V7jMsV4utd65PJv"
        }
        self.max_retries = 3

    async def _request_with_retry(self, client_method, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                response = await client_method(*args, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 408 and attempt < self.max_retries - 1:
                    logger.warning(f"Torob API returned 408. Retrying ({attempt + 1}/{self.max_retries - 1})...")
                    await asyncio.sleep(1)  # simple backoff
                    continue
                raise
        return None

    async def send_frame(self, frame_path: Path, text_prompt: str) -> Optional[str]:
        """Sends a frame to the external detection service and returns the base64 encoded image if successful."""
        url = "https://inell-threadless-thirdly.ngrok-free.dev/detect"

        try:
            async with httpx.AsyncClient() as client:
                with open(frame_path, "rb") as f:
                    files = {"image": (frame_path.name, f, "image/jpeg")}
                    data = {"text_prompt": text_prompt}

                    logger.info(f"Sending frame to {url} with prompt: '{text_prompt}'")
                    response = await client.post(url, files=files, data=data, timeout=40)
                    response.raise_for_status()

                    response_data = response.json()

                    if isinstance(response_data, list) and len(response_data) > 0:
                        best_match = response_data[0]
                        if "image_base64" in best_match:
                            logger.info("Successfully received base64 image from detection service.")
                            return best_match["image_base64"]
                        else:
                            logger.warning("Detection successful, but 'image_base64' not found in the response.")
                            return None
                    elif isinstance(response_data, dict) and "error" in response_data:
                        logger.error(f"Error from detection service: {response_data['error']}")
                        return None
                    else:
                        logger.warning(f"Detection service returned an empty or unexpected response: {response_data}")
                        return None

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while sending frame to detection service: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending frame: {e}", exc_info=True)
            return None
        
    async def convert_file_to_base64_async(self, file_path: Path) -> Optional[str]:
        try:
            async with aiofiles.open(file_path, "rb") as f:
                image_bytes = await f.read()
                base64_bytes = base64.b64encode(image_bytes)
                return base64_bytes.decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            logger.error(f"An error occurred during async base64 conversion: {e}", exc_info=True)
            return None

    async def upload_to_torob(self, image_base64: str) -> Optional[str]:
        """Uploads a base64 encoded image to Torob's image upload API."""
        torob_url = "https://api.torob.com/v4/base-product/search-image-upload/"

        try:
            image_data = base64.b64decode(image_base64)
            files = {'img': ('image.jpg', image_data, 'image/jpeg')}

            async with httpx.AsyncClient() as client:
                logger.info("Uploading image to Torob...")
                response = await self._request_with_retry(client.post, torob_url, files=files, headers=self.torob_headers, timeout=40)
                if not response: return None

                response_data = response.json()
                image_url = response_data.get("image_url")

                if image_url:
                    logger.info(f"Successfully uploaded image to Torob. Image URL: {image_url}")
                    return image_url
                else:
                    logger.error(f"Torob upload API did not return an 'image_url'. Response: {response_data}")
                    return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while uploading to Torob: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Torob upload: {e}", exc_info=True)
            return None

    async def search_on_torob(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Searches Torob by image URL and returns all parsed product information,
        along with the URL of the image that was searched.
        """
        torob_url = f"https://api.torob.com/v4/base-product/search-by-image/?image_url={image_url}"

        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Searching on Torob with image URL: {image_url}")
                response = await self._request_with_retry(client.get, torob_url, headers=self.torob_headers, timeout=40)
                if not response: return None

                results = response.json()

                if results.get("results") and len(results["results"]) > 0:
                    all_results = results["results"]
                    products_info = [
                        {
                            "name": result.get("name1"),
                            "link": f"https://torob.com{result.get('web_client_absolute_url')}",
                            "random_key": result.get("random_key"),
                            "rank": i + 1
                        }
                        for i, result in enumerate(all_results)
                    ]
                    logger.info(f"Successfully found {len(products_info)} products on Torob.")

                    return {
                        "products": products_info,
                        "searched_image_url": results.get("uploaded_image_url")
                    }
                else:
                    logger.warning("No products found on Torob for the given image.")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during Torob search: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Torob search: {e}", exc_info=True)
            return None

    async def search_on_torob_by_text(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Searches Torob by a text query and returns up to 10 non-advertisement products, including their image URLs."""
        search_url = f"https://api.torob.com/v4/base-product/search/?q={query}&size=100&page=1"

        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Searching on Torob with text query: '{query}'")
                response = await self._request_with_retry(client.get, search_url, headers=self.torob_headers, timeout=40)
                if not response: return None

                data = response.json()
                results = data.get("results", [])

                if not results:
                    logger.warning(f"No text search results found on Torob for query: '{query}'")
                    return None

                products_info = []
                for result in results:
                    if not result.get("is_adv"):
                        products_info.append({
                            "name": result.get("name1"),
                            "link": f"https://torob.com{result.get('web_client_absolute_url')}",
                            "random_key": result.get("random_key"),
                            "image_url": result.get("image_url")
                        })
                        if len(products_info) >= 10:
                            break

                logger.info(f"Found {len(products_info)} non-advertisement products from text search.")
                return products_info

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during Torob text search: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Torob text search: {e}", exc_info=True)
            return None

    async def search_product(self, frame_path: Path, text_prompt: str) -> Optional[Dict[str, Any]]:
        image_base64 = await self.convert_file_to_base64_async(frame_path)
        if not image_base64:
            logger.error("Failed to get base64 image from detection service. Aborting product search.")
            return None

        image_url = await self.upload_to_torob(image_base64)
        if not image_url:
            logger.error("Failed to upload image to Torob. Aborting product search.")
            return None

        product_info = await self.search_on_torob(image_url)
        if not product_info:
            logger.warning("Failed to find product information on Torob.")
            return None

        return product_info

product_service = ProductService()
