from typing import Dict, Any, BinaryIO
from azure.storage.blob import BlobServiceClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from io import BytesIO
from .base import CloudProvider

class AzureProvider(CloudProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blob_service_client = BlobServiceClient.from_connection_string(
            config['connection_string']
        )
        self.container_client = self.blob_service_client.get_container_client(
            config['container_name']
        )
        self.ai_client = TextAnalyticsClient(
            endpoint=config['ai_endpoint'],
            credential=AzureKeyCredential(config['ai_key'])
        )
        
    async def upload_file(self, file: BinaryIO, path: str) -> str:
        blob_client = self.container_client.get_blob_client(path)
        await blob_client.upload_blob(file, overwrite=True)
        return f"https://{self.config['account_name']}.blob.core.windows.net/{self.config['container_name']}/{path}"
    
    async def download_file(self, path: str) -> BinaryIO:
        blob_client = self.container_client.get_blob_client(path)
        stream = BytesIO()
        await blob_client.download_blob().readinto(stream)
        stream.seek(0)
        return stream
    
    async def get_vector_store_client(self):
        return self.ai_client