"""The team of agents.

state.py     the shared data passed between agents
llm.py       one configured connection to the Groq LLM
router.py    decides which specialist handles a message
greeting.py  replies to greetings and small talk
clarify.py   asks one short follow-up when a request is vague
answer.py    answers policy questions from the knowledge base (stub until Phase 4)
summarise.py summarises notes and lists action items
graph.py     wires the router to the specialists
"""
