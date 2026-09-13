"""Everything about the knowledge base the assistant answers from.

embeddings.py    the model that turns text into vectors (BGE-M3)
chunking.py      split a document into overlapping passages
vector_store.py  opens / creates the Chroma index
build_index.py   the offline job: read documents -> split -> embed -> store
search.py        question -> the closest passages (used by the answer agent)
"""
