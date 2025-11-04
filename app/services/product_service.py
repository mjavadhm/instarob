import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
from httpx_socks import AsyncProxyTransport

from app.core.logging import get_logger

logger = get_logger()

class ProductService:
    def __init__(self):
        pass


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
                        # Assuming the first item is the one with the highest score as per the documentation
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

    async def upload_to_torob(self, image_base64: str) -> Optional[str]:
        """Uploads a base64 encoded image to Torob's image upload API."""
        torob_url = "https://api.torob.com/v4/base-product/search-image-upload/"
        transport = AsyncProxyTransport.from_url("socks5://127.0.0.1:2444")

        try:
            image_data = base64.b64decode(image_base64)
            files = {'img': ('image.jpg', image_data, 'image/jpeg')}

            async with httpx.AsyncClient(transport=transport) as client:
                logger.info("Uploading image to Torob...")
                response = await client.post(torob_url, files=files, timeout=40)
                response.raise_for_status()

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
        Searches Torob by image URL and returns parsed product information for the top 5 results,
        along with the URL of the image that was searched.
        """
        torob_url = f"https://api.torob.com/v4/base-product/search-by-image/?image_url={image_url}"
        transport = AsyncProxyTransport.from_url("socks5://127.0.0.1:2444")

        try:
            async with httpx.AsyncClient(transport=transport) as client:
                logger.info(f"Searching on Torob with image URL: {image_url}")
                response = await client.get(torob_url, timeout=40)
                response.raise_for_status()

                results = response.json()

                if results.get("results") and len(results["results"]) > 0:
                    top_results = results["results"][:5]
                    products_info = [
                        {
                            "name": result.get("name1"),
                            "link": f"https://torob.com{result.get('web_client_absolute_url')}",
                            "random_key": result.get("random_key")
                        }
                        for result in top_results
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

    async def search_product(self, frame_path: Path, text_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the full product search workflow:
        1. Sends a frame for detection.
        2. Uploads the resulting base64 image to Torob.
        3. Searches Torob with the uploaded image URL.
        """
        # 1. Get the base64 encoded image from the detection service
        image_base64 = await self.send_frame(frame_path, text_prompt)
        if not image_base64:
            logger.error("Failed to get base64 image from detection service. Aborting product search.")
            return None

        # 2. Upload the image to Torob
        image_url = await self.upload_to_torob(image_base64)
        if not image_url:
            logger.error("Failed to upload image to Torob. Aborting product search.")
            return None

        # 3. Search for the product on Torob
        product_info = await self.search_on_torob(image_url)
        if not product_info:
            logger.warning("Failed to find product information on Torob.")
            return None

        return product_info

product_service = ProductService()
