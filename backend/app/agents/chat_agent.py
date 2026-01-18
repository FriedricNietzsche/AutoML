"""
Chat Agent - LangChain integration for handling user chat messages.

This agent processes natural language messages from users and can:
- Answer questions about the current project
- Provide suggestions for improving the model
- Explain what's happening at each stage
- Route requests to appropriate actions (retraining, parameter changes, etc.)
"""
from typing import Dict, Any, Optional, List
import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm.langchain_client import get_openrouter_llm
from app.events.schema import StageID

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """Structured response from the chat agent."""
    
    message: str = Field(description="The assistant's response message to the user")
    intent: str = Field(description="Detected intent: 'question', 'suggestion', 'command', 'acknowledgment'")
    suggested_action: Optional[str] = Field(
        default=None, 
        description="Suggested action if any: 'retrain', 'adjust_params', 'change_model', 'export', 'none'"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Suggested parameters for the action if applicable"
    )


class ChatAgent:
    """
    LangChain-powered chat agent for conversational AI assistance.
    
    Provides context-aware responses about the AutoML pipeline and can
    suggest actions based on user requests.
    """
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7):
        """
        Initialize the chat agent.
        
        Args:
            model: OpenRouter model to use (e.g., 'anthropic/claude-3.5-sonnet')
            temperature: Temperature for response generation (0.0-1.0)
        """
        self.llm = get_openrouter_llm(model=model, temperature=temperature)
        self.conversation_history: Dict[str, List[Any]] = {}  # project_id -> messages
        
        # System prompt that defines the agent's role
        self.system_prompt = """You are an AI assistant helping users build machine learning models through an AutoML platform.

Your role:
- Answer questions about the ML pipeline, current stage, and model performance
- Provide helpful suggestions for improving model accuracy, speed, or interpretability
- Explain technical concepts in a clear, concise way
- Detect when users want to take actions (retrain, change parameters, export, etc.)
- Be encouraging and supportive

Current context:
- Stage: {current_stage}
- Task Type: {task_type}
- Target: {target_column}
- Model: {current_model}
- Status: {stage_status}

Guidelines:
- Keep responses concise (1-3 sentences typically)
- Use clear, non-technical language when possible
- When suggesting actions, be specific about parameters
- If unsure, ask clarifying questions

Respond with:
- message: Your response to the user
- intent: Classify as 'question', 'suggestion', 'command', or 'acknowledgment'
- suggested_action: If user wants action, specify: 'retrain', 'adjust_params', 'change_model', 'export', or 'none'
- parameters: Any specific parameters for the action (e.g., {{"learning_rate": 0.01}})
"""
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_message}"),
        ])
        
        # Parser for structured output
        self.parser = JsonOutputParser(pydantic_object=ChatResponse)
        
    async def process_message(
        self,
        project_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Process a user chat message and generate a response.
        
        Args:
            project_id: The project ID for conversation context
            user_message: The user's message
            context: Current project context (stage, model, etc.)
            
        Returns:
            ChatResponse with message, intent, and suggested actions
        """
        # Default context if none provided
        if context is None:
            context = {
                "current_stage": "PARSE_INTENT",
                "task_type": "unknown",
                "target_column": "unknown",
                "current_model": "not selected",
                "stage_status": "IN_PROGRESS"
            }
        
        # LangChain path if LLM available
        if self.llm:
            try:
                # Get conversation history for this project
                history = self.conversation_history.get(project_id, [])
                
                # Build the chain
                chain = self.prompt | self.llm | self.parser
                
                # Invoke the chain
                result = await chain.ainvoke({
                    "current_stage": context.get("current_stage", "unknown"),
                    "task_type": context.get("task_type", "unknown"),
                    "target_column": context.get("target_column", "unknown"),
                    "current_model": context.get("current_model", "not selected"),
                    "stage_status": context.get("stage_status", "IN_PROGRESS"),
                    "history": history[-10:],  # Last 10 messages for context
                    "user_message": user_message,
                })
                
                # Update conversation history
                if project_id not in self.conversation_history:
                    self.conversation_history[project_id] = []
                    
                self.conversation_history[project_id].append(HumanMessage(content=user_message))
                self.conversation_history[project_id].append(AIMessage(content=result["message"]))
                
                # Convert to ChatResponse object
                return ChatResponse(**result)
                
            except Exception as e:
                logger.warning(f"LangChain chat failed: {e}, using fallback")
                return self._fallback_response(user_message, context)
        
        # Fallback if no LLM
        return self._fallback_response(user_message, context)
    
    def _fallback_response(self, user_message: str, context: Dict[str, Any]) -> ChatResponse:
        """
        Generate a simple rule-based response when LLM is unavailable.
        
        Args:
            user_message: The user's message
            context: Current project context
            
        Returns:
            ChatResponse with basic response
        """
        msg_lower = user_message.lower()
        
        # Detect common intents
        if any(word in msg_lower for word in ["?", "what", "how", "why", "explain"]):
            intent = "question"
            message = f"I understand you have a question. Currently at {context.get('current_stage', 'unknown')} stage. The system is processing your request."
            
        elif any(word in msg_lower for word in ["retrain", "train again", "rerun"]):
            intent = "command"
            message = "I'll initiate retraining with your requested changes."
            return ChatResponse(
                message=message,
                intent=intent,
                suggested_action="retrain",
                parameters={}
            )
            
        elif any(word in msg_lower for word in ["export", "download", "save"]):
            intent = "command"
            message = "I'll prepare the export bundle for you."
            return ChatResponse(
                message=message,
                intent=intent,
                suggested_action="export",
                parameters={}
            )
            
        elif any(word in msg_lower for word in ["change", "switch", "use", "try"]):
            intent = "suggestion"
            message = "I've noted your suggestion. You can adjust parameters after the current stage completes."
            
        else:
            intent = "acknowledgment"
            message = "Thanks for your input! I'm processing your request."
        
        return ChatResponse(
            message=message,
            intent=intent,
            suggested_action="none",
            parameters=None
        )
    
    def clear_history(self, project_id: str) -> None:
        """Clear conversation history for a project."""
        if project_id in self.conversation_history:
            del self.conversation_history[project_id]
    
    def get_history(self, project_id: str) -> List[Any]:
        """Get conversation history for a project."""
        return self.conversation_history.get(project_id, [])


# Global chat agent instance
_chat_agent: Optional[ChatAgent] = None


def get_chat_agent(model: Optional[str] = None, temperature: float = 0.7) -> ChatAgent:
    """
    Get or create the global chat agent instance.
    
    Args:
        model: OpenRouter model to use
        temperature: Temperature for response generation
        
    Returns:
        ChatAgent instance
    """
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent(model=model, temperature=temperature)
    return _chat_agent
