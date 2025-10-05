from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage
import json
import google.cloud.aiplatform as vertex_ai
from google.cloud.aiplatform.prediction_service import PredictionServiceClient
from azure.ai.textanalytics import TextAnalyticsClient
import boto3

class CloudAIService(ABC):
    @abstractmethod
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        pass
    
    @abstractmethod
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class AWSAIService(CloudAIService):
    def __init__(self, bedrock_client):
        self.client = bedrock_client
        
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        formatted_messages = "\n\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        response = self.client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": formatted_messages,
                "max_tokens": 1000,
                "temperature": 0.7
            })
        )
        return json.loads(response['body'].read())['completion']
        
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps({
                "inputText": texts
            })
        )
        return json.loads(response['body'].read())['embedding']

class AzureAIService(CloudAIService):
    def __init__(self, ai_client: TextAnalyticsClient):
        self.client = ai_client
        
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        formatted_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        response = await self.client.analyze_sentiment(
            documents=[formatted_messages],
            show_opinion_mining=True
        )
        # Using Azure OpenAI for completion
        deployment_name = "gpt-4"  # Configure based on your deployment
        response = await self.client.completions.create(
            model=deployment_name,
            prompt=formatted_messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].text.strip()
        
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Using Azure OpenAI embeddings
        deployment_name = "text-embedding-ada-002"  # Configure based on your deployment
        responses = []
        for text in texts:
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=text
            )
            responses.append(response.data[0].embedding)
        return responses

class GCPAIService(CloudAIService):
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.client = PredictionServiceClient()
        vertex_ai.init(project=project_id, location=location)
        
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        formatted_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        
        # Using Vertex AI PaLM API
        model = vertex_ai.TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
            formatted_messages,
            temperature=0.7,
            max_output_tokens=1000
        )
        return response.text
        
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Using Vertex AI embeddings
        model = vertex_ai.TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        embeddings = []
        
        for text in texts:
            response = model.get_embeddings(text)
            embeddings.append(response.values)
            
        return embeddings

# Factory for creating cloud-specific AI services
class CloudAIServiceFactory:
    @staticmethod
    def create_service(provider: str, config: Dict[str, Any]) -> CloudAIService:
        if provider == "aws":
            bedrock_client = boto3.client('bedrock-runtime')
            return AWSAIService(bedrock_client)
        
        elif provider == "azure":
            ai_client = TextAnalyticsClient(
                endpoint=config['ai_endpoint'],
                credential=config['ai_key']
            )
            return AzureAIService(ai_client)
        
        elif provider == "gcp":
            return GCPAIService(
                project_id=config['project_id'],
                location=config['location']
            )
        
        raise ValueError(f"Unsupported cloud provider: {provider}")