"""Training and evaluation data: corpus → topic neighbors → synthetic facts → questions → verification.

Ported from the nvme-llm research scripts (data114_docs, data114_neighbors, gen114_facts, gen118_questions, gen124_verify)
with identical prompts, parsing, filtering and file formats. README.md in this directory has the commands that rebuild the
released training data and the Qdrant evaluation set.
"""
