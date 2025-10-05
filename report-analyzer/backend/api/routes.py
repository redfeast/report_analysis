from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..core.services.analysis_service import AnalysisService
from ..core.services.cloud_ai_service import CloudAIServiceFactory
from ..config.settings import Settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_analysis_service(settings: Settings = Depends(Settings)) -> AnalysisService:
    try:
        # Initialize cloud AI service
        cloud_ai = CloudAIServiceFactory.create_service(
            provider=settings.cloud_provider,
            config=settings.cloud_config
        )
        
        # Initialize service with configurations
        return AnalysisService(
            config=settings.config,
            tools=settings.tools_config,
            vector_store_config=settings.vector_store_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service initialization failed: {str(e)}")

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    service: AnalysisService = Depends(get_analysis_service)
):
    return await service.process_document(file)

@app.post("/analyze")
async def analyze_query(
    query: str,
    doc_id: str,
    service: AnalysisService = Depends(get_analysis_service)
):
    return await service.process_query(query, doc_id)