from typing import List
import uuid
from app.models.reel_in import ReelIn
from app.models.reel_out import ReelOut
from app.core.config import settings
from app.services.storage import storage
from app.core.logging import get_logger, request_id_var

logger = get_logger()

class ReelsService:
    async def process_reel(self, reel_in: ReelIn) -> ReelOut:
        request_id = request_id_var.get()

        if reel_in.request_id:
            existing_keys = await storage.get(reel_in.request_id)
            if existing_keys:
                logger.info(f"Returning existing keys for request_id: {reel_in.request_id}")
                return ReelOut(request_id=request_id, keys=existing_keys)

        keys = [uuid.uuid4() for _ in range(settings.REEL_KEYS_COUNT)]

        if reel_in.request_id:
            await storage.set(reel_in.request_id, keys)

        logger.info(f"Generated {len(keys)} new keys for request_id: {request_id}")

        return ReelOut(request_id=request_id, keys=keys)

reels_service = ReelsService()
