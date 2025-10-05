from typing import Dict, Any, BinaryIO
from google.cloud import storage
from google.cloud import aiplatform
from infra.cloud import CloudProvider

class GCPProvider(CloudProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(config['bucket'])
        aiplatform.init(
            project=config['project_id'],
            location=config['location']
        )
        
    async def upload_file(self, file: BinaryIO, path: str) -> str:
        blob = self.bucket.blob(path)
        blob.upload_from_file(file)
        return f"gs://{self.config['bucket']}/{path}"
    
    async def download_file(self, path: str) -> BinaryIO:
        blob = self.bucket.blob(path)
        file_obj = BinaryIO()
        blob.download_to_file(file_obj)
        file_obj.seek(0)
        return file_obj
    
    async def get_vector_store_client(self):
        return aiplatform.VertexAI()