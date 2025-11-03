from typing import Dict, Any, List
from app.core.logging import get_logger

logger = get_logger()

class Storage:
    _data: Dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        logger.info(f"Getting data for key: {key}")
        return self._data.get(key)

    async def set(self, key: str, value: Any):
        logger.info(f"Setting data for key: {key}")
        self._data[key] = value

storage = Storage()
