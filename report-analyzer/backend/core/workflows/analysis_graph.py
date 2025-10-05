from typing import Dict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import Graph, StateGraph
from pydantic import BaseModel
from ..services.cloud_ai_service import CloudAIServiceFactory

class GraphState(BaseModel):
    messages: List[BaseMessage]
    context: Dict = {}
    current_step: str = "START"
    ai_response: Dict[str, Any] = {}

class AnalysisGraph:
    def __init__(self, config: Dict[str, Any], tools: Dict[str, Any], vector_store):
        # Initialize cloud AI service using factory
        self.cloud_ai = CloudAIServiceFactory.create_service(
            provider=config['cloud_provider'],
            config=config['ai_config']
        )
        self.tools = tools
        self.vector_store = vector_store
        self.graph = self._create_workflow()

    async def _process_with_ai(self, state: GraphState) -> GraphState:
        """Process messages with cloud AI and update state"""
        try:
            # Generate AI response
            ai_response = await self.cloud_ai.generate_response(state.messages)
            state.context['ai_response'] = ai_response

            # Check if embeddings are needed
            if "embedding" in state.messages[-1].content.lower() or state.context.get('needs_embeddings'):
                embeddings = await self.cloud_ai.create_embeddings(
                    [msg.content for msg in state.messages]
                )
                state.context['embeddings'] = embeddings

            return state
        except Exception as e:
            state.context['error'] = str(e)
            return state

    def _create_workflow(self) -> Graph:
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("document_search", self._document_search)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("analyze", self._analyze_needs)
        workflow.add_node("ai_process", self._process_with_ai)
        
        # Define edges with conditions
        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._analyze_needs,
            {
                "DOC_ONLY": "document_search",
                "NEEDS_WEB": "web_search"
            }
        )
        workflow.add_edge("document_search", "ai_process")
        workflow.add_edge("web_search", "document_search")
        
        return workflow.compile()

    def _analyze_needs(self, state: GraphState) -> str:
        """Determine if web search is needed"""
        last_message = state.messages[-1].content.lower()
        if any(keyword in last_message for keyword in ["compare", "external", "web", "internet"]):
            return "NEEDS_WEB"
        return "DOC_ONLY"

    async def _document_search(self, state: GraphState) -> GraphState:
        """Perform document search using vector store"""
        query = state.messages[-1].content
        docs = await self.vector_store.asimilarity_search(query)
        state.context["doc_results"] = docs
        return state

    async def _web_search(self, state: GraphState) -> GraphState:
        """Perform web search using provided tools"""
        query = state.messages[-1].content
        search_results = await self.tools["web_search"].arun(query)
        state.context["web_results"] = search_results
        return state