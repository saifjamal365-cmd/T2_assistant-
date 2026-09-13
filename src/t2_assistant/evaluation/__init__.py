"""Measuring quality, not just building the thing.

    dataset.py    load the 500-question evaluation set (eval/dataset.json)
    retrieval.py  fast check: is the right document in the search results?
    quality.py    full-pipeline check: does chat() answer correctly and honestly?
    report.py     turn both into numbers, write a report, log to MLflow
    run.py        the command that runs all of it: python -m t2_assistant.evaluation.run

The dataset itself is built by scripts/build_eval_set.py from the knowledge-base
generator's own source data - not hand-typed - so grading against it is fair.
"""
