#module that handles aggregate queries to avoid calling the LLM
import pandas as pd
import os
import glob 

AGGREGATE_KEYWORDS = ["total", "sum", "average", "avg",
"mean", "how many", "number of","highest", "lowest",
"count", "maximum", "minimum", "max", "min"]

def is_aggregate_query(query): -> bool:
    return any(keyword in query.lower() for keyword in AGGREGATE_KEYWORDS)  


def run_pandas_query(query:str) -> str:
    if not csv_file_paths:
        return "No CSV files loaded. Please upload a CSV file to perform analysis."

    