"""The shared state that flows through the agent graph.

LangGraph passes one dictionary from node to node. Each node reads what it needs
and returns the fields it wants to change. We keep it small:

    messages       the whole conversation (user + assistant turns)
    route          which specialist the router chose
    route_reason   the router's one-line reason (shown in the trace / API)
    answer_model   the Groq model id the chosen specialist writes the reply
                    with, already resolved before the graph runs

Named `answer_model`, not `model`: LangGraph hands this whole dictionary to
every node, including the router - which never reads this field, since
routing always runs on a fixed model (see graph.py's _router_node). A plain
`model` key showing up in the router's own traced inputs reads as "the model
the router used," which is exactly wrong; the specific name makes clear what
it actually is even where it's not consumed.

`messages` uses LangGraph's `add_messages` reducer: when a node returns new
messages, they are appended to the list rather than replacing it.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["greeting", "answer", "summarise", "clarify"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route | None
    route_reason: str | None
    answer_model: str
