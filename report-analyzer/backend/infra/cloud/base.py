from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Any

class CloudProvider(ABC):
    @abstractmethod
    async def upload_file(self, file: BinaryIO, path: str) -> str:
        pass
    
    @abstractmethod
    async def download_file(self, path: str) -> BinaryIO:
        pass
    
    @abstractmethod
    async def get_vector_store_client(self) -> Any:
        pass