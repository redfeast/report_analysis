from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from core.workflows import GraphState, AnalysisGraph

class AnalysisService:
    def __init__(self, llm, tools, vector_store):
        self.llm = llm
        self.tools = tools
        self.vector_store = vector_store
        self.workflow = AnalysisGraph(tools, llm, vector_store)
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent report analyzer that helps users understand documents and compare them with external information. 
            When analyzing documents, always:
            1. Consider the context from both document search and web search
            2. Provide clear citations for your sources
            3. Highlight key insights and comparisons"""),
            ("user", "{input}"),
            ("context", "{context}"),
            ("assistant", "I'll analyze that based on the available information."),
        ])

        agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def process_document(self, file_content: str) -> Dict[str, Any]:
        """Process and index a new document."""
        try:
            # Add document to vector store
            docs = await self.vector_store.aadd_documents([file_content])
            return {
                "status": "success",
                "message": "Document processed successfully",
                "doc_id": docs[0].id if docs else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}"
            }

    async def process_query(self, query: str, conversation_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """Process a user query using the workflow and agent."""
        try:
            # Initialize state
            messages = conversation_history or []
            messages.append(HumanMessage(content=query))
            
            initial_state = GraphState(
                messages=messages,
                context={},
                current_step="START"
            )

            # Run workflow
            workflow_result = await self.workflow.graph.ainvoke(initial_state)

            # Prepare context for agent
            context = self._prepare_context(workflow_result.context)

            # Get agent response
            agent_response = await self.agent.ainvoke({
                "input": query,
                "context": context
            })

            return {
                "status": "success",
                "response": agent_response["output"],
                "sources": {
                    "document_sources": workflow_result.context.get("doc_results", []),
                    "web_sources": workflow_result.context.get("web_results", [])
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    def _prepare_context(self, workflow_context: Dict) -> str:
        """Prepare context from workflow results for the agent."""
        context_parts = []
        
        if "doc_results" in workflow_context:
            context_parts.append("From documents:")
            for i, doc in enumerate(workflow_context["doc_results"], 1):
                context_parts.append(f"{i}. {doc.page_content}")

        if "web_results" in workflow_context:
            context_parts.append("\nFrom web search:")
            context_parts.append(workflow_context["web_results"])

        return "\n".join(context_parts)