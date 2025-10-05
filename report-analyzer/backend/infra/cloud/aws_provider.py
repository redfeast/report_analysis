import boto3
from .base import CloudProvider

class AWSProvider(CloudProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime')
        
    async def upload_file(self, file: BinaryIO, path: str) -> str:
        bucket = self.config['bucket']
        self.s3_client.upload_fileobj(file, bucket, path)
        return f"s3://{bucket}/{path}"
    
    async def get_vector_store_client(self):
        return self.bedrock_client