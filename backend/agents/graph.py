from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from agents.state import AgentState
from agents.agents import suggest_roles_node, crawl_jobs_node, fill_form_node, submit_application_node

def build_discovery_graph():
    """Graph for suggesting roles and crawling jobs"""
    workflow = StateGraph(AgentState)
    workflow.add_node("suggest_roles", suggest_roles_node)
    workflow.add_node("crawl_jobs", crawl_jobs_node)
    
    workflow.set_entry_point("suggest_roles")
    workflow.add_edge("suggest_roles", "crawl_jobs")
    workflow.add_edge("crawl_jobs", END)
    
    return workflow.compile()

def should_interrupt(state: AgentState) -> Literal["human_approval", "submit_application"]:
    if state.get("requires_human_approval"):
        return "human_approval"
    return "submit_application"

def human_approval_node(state: AgentState) -> dict:
    """A dummy node that acts as a pause point. The state will be updated externally before resuming."""
    pass

def build_application_graph():
    """Graph for filling job forms with human-in-the-loop"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("fill_form", fill_form_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("submit_application", submit_application_node)
    
    workflow.set_entry_point("fill_form")
    
    workflow.add_conditional_edges(
        "fill_form",
        should_interrupt,
        {
            "human_approval": "human_approval",
            "submit_application": "submit_application"
        }
    )
    
    workflow.add_edge("human_approval", "submit_application")
    workflow.add_edge("submit_application", END)
    
    # We use MemorySaver so we can retain state when interrupted
    checkpointer = MemorySaver()
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"]
    )
    
    return app
