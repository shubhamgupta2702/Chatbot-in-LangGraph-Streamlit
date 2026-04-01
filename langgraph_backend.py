from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tool_node
from langchain_community.tools import DuckDuckGoSearchRun
load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-Next",
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  
def chat(state:ChatState):
  message = state['messages']
  
  response = model.invoke(message).content
  
  return {"messages":response}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_model", chat)

graph.add_edge(START, "chat_model")
graph.add_edge("chat_model", END)


chatbot = graph.compile(checkpointer=checkpointer)