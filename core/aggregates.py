#module that handles aggregate queries to avoid calling the LLM
import pandas as pd
import os
import glob 

#configuration list
#built once and used to check if a query is aggregate in nature
AGGREGATE_KEYWORDS = ["total", "sum", "average", "avg",
"mean", "how many", "number of","highest", "lowest",
"count", "maximum", "minimum", "max", "min", "percent", 
"percentage", "median", "mode", "std", "standard deviation",
"compare", "comparison", "trend", "trends", "distribution",]

AGGREGATE_DISPATCH = {
    "sum":    "sum",
    "total":  "sum",
    "average": "mean",
    "avg":    "mean",
    "mean":   "mean",
    "max":    "max",
    "maximum": "max",
    "highest": "max",
    "min":    "min",
    "minimum": "min",
    "lowest": "min",
    "median": "median",
    "count":  "count",
    "how many": "count",
}

def is_aggregate_query(query:str) -> bool:
    #any checks for keywords in query, if finds stops and returns True
    return any(keyword in query.lower() for keyword in AGGREGATE_KEYWORDS)  


def run_pandas_query(query:str, tabular_file_paths:list) -> str:
    operations = []
    for keyword, agg_func in AGGREGATE_DISPATCH.items():
        if keyword in query.lower():
            if agg_func not in operations:
                operations.append(agg_func)
    
    target_files = []
    for file in tabular_file_paths:
        file_name = os.path.splitext(os.path.basename(file).lower())[0]
        if file_name in query.lower():
            target_files.append(file)
    if not target_files:
        target_files = tabular_file_paths

    dataframes = files_to_dfs(target_files)

    target_columns = []
    for df in dataframes:
        for col in df.columns:
            if col.lower() in query.lower():
                target_columns.append(col)

def files_to_dfs(tabular_file_paths:list) -> list:
    dataframes = []
    for file in tabular_file_paths:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        dataframes.append(df)
    return dataframes