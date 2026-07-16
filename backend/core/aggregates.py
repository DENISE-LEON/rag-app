import os
import re
import json
from typing import Any, Optional, List
import pandas as pd
from langchain_anthropic import ChatAnthropic
import backend.config as cfg
import operator

llm = cfg.llm

AGGREGATE_KEYWORDS = [
    "total",
    "sum",
    "average",
    "avg",
    "mean",
    "how many",
    "number of",
    "highest",
    "lowest",
    "count",
    "maximum",
    "minimum",
    "max",
    "min",
    "percent",
    "percentage",
    "median",
    "mode",
    "std",
    "standard deviation",
]

SUMMARY_KEYWORDS = ["summarize", "summary", "overview", "describe", "description"]
FILTER_KEYWORDS = ["filter", "where", "select", "subset", "condition"]

AGGREGATE_DISPATCH = {
    "sum": "sum",
    "total": "sum",
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "max": "max",
    "maximum": "max",
    "highest": "max",
    "min": "min",
    "minimum": "min",
    "lowest": "min",
    "median": "median",
    "count": "count",
    "how many": "count",
    "number of": "count",
}

SUPPORTED_AGGREGATES = set(AGGREGATE_DISPATCH.values())

FILTER_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

OPERATOR_DISPATCH = {
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "=": "==",
    "==": "==",
    "!=": "!=",
    "less": "<",
    "fewer": "<",
    "below": "<",
    "under": "<",
    "greater": ">",
    "more": ">",
    "above": ">",
    "over": ">",
    "equal": "==",
    "equals": "==",
    "equal to": "==",
}




def pandas_pipeline(query: str, tabular_files: List[Any]) -> str:
    if not query or not tabular_files:
        return "No tabular data was provided to analyze."

    normalized_query = query.lower().strip()

    dataframes = _collect_dataframes(tabular_files)
    if not dataframes:
        return "No tabular data could be loaded from the provided files."

    filter_spec = _parse_filter(normalized_query)
    aggregate_op = _parse_aggregate(normalized_query)
    target_column = None

    intent = _route_intent(normalized_query, filter_spec, aggregate_op)

    if intent == "unknown":
        metadata_hint = _build_tabular_metadata_hint(tabular_files)
        smart_plan = _smart_interpret_query(normalized_query, metadata_hint)

        if smart_plan is not None:
            filter_spec = smart_plan.get("filter")
            aggregate_op = smart_plan.get("aggregate")
            target_column = smart_plan.get("target_column")
            intent = _route_intent(normalized_query, filter_spec, aggregate_op)

    if intent == "unknown":
        return "unable to answer"

    results: List[str] = []
 
    for df in dataframes:
        label = os.path.splitext(
            os.path.basename(getattr(df, "attrs", {}).get("source_name", "table"))
        )[0]
        current_target_column = target_column
        if current_target_column is None:
            current_target_column = _select_target_column(normalized_query, df, filter_spec)

        if intent == "filter":
            filtered = _apply_filter(df, filter_spec)
            if filtered is None:
                results.append(f"Column '{filter_spec['column']}' was not found in {label}.")
            else:
                results.append(
                    f"{label}: {len(filtered)} rows matched the filter out of {len(df)} total rows."
                )

        elif intent == "aggregate":
            msg = _aggregate_handler(
                df = df,
                aggregate_op=aggregate_op,
                label=label,
                target_column=current_target_column,
            )
            results.append(msg)

        elif intent == "mixed":
            msg = _mixed_handler(
                df=df,
                filter_spec=filter_spec,
                aggregate_op=aggregate_op,
                label=label,
                target_column=current_target_column,
            )
            results.append(msg)
#### add summarize intent 

    if not results:
        return "No results could be computed from the provided data."

    return "\n".join(results)

#return summary of all dfs
def build_pandas_summary(tabular_files: List[Any]) -> str:
    summaries: List[str] = []

    for entry in tabular_files or []:
        if isinstance(entry, dict):
            filename = entry.get("filename", "table")
            for df in entry.get("dataframes") or []:
                if isinstance(df, pd.DataFrame):
                    summaries.append(_summarize_dataframe(df, filename))
        elif isinstance(entry, pd.DataFrame):
            summaries.append(_summarize_dataframe(entry, "dataframe"))

    return "\n".join(summaries) if summaries else "No tabular data available."

#return summary of rows & columns for df
def _summarize_dataframe(df: pd.DataFrame, source_name: str) -> str:
    summary_bits = [f"{source_name}: {len(df)} rows, {len(df.columns)} columns. \n columns: {df.columns}"]

    numeric_cols = _numeric_columns(df)
    if numeric_cols:
        for col in numeric_cols[:3]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                summary_bits.append(
                    f"{col}: min={series.min()}, max={series.max()}, mean={series.mean():.2f}"
                )

    return " | ".join(summary_bits)


def _build_tabular_metadata_hint(tabular_files: List[Any]) -> str:
    hints: List[str] = []

    for file in tabular_files or []:
        if not isinstance(file, dict):
            continue

        filename = file.get("filename", "unknown")
        columns_list = file.get("columns", [])
        row_counts = file.get("row_counts", [])

        for i, cols in enumerate(columns_list):
            cols_text = ", ".join(cols[:12]) if cols else "no columns detected"
            row_count = row_counts[i] if i < len(row_counts) else "unknown"
            hints.append(
                f"File: {filename} | Table {i + 1} | Rows: {row_count} | Columns: {cols_text}"
            )

    return "\n".join(hints)

#send query to llm, have it parse the query and return the filters, columns, aggregates
def _smart_interpret_query(
    normalized_query: str, metadata_hint
) -> Optional[dict[str, Any]]:
    if not normalized_query:
        return None

    prompt = f"""
You are converting a natural-language table question into a JSON plan.

Return ONLY valid JSON with this exact shape:
{{
  "aggregate": "sum" | "mean" | "max" | "min" | "median" | "count" | null,
  "target_column": string | null,
  "filter": {{
    "column": string,
    "op": "<" | "<=" | ">" | ">=" | "==" | "!=",
    "value": number
  }} | null
}}

Rules:
- Do not explain anything.
- Do not include markdown.
- If no aggregate is clearly requested, use null.
- If no target column is clear, use null.
- If no filter is present, use null.
- Use only likely column names from the metadata when possible.

Tabular metadata:
{metadata_hint}

Question:
{normalized_query}
"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        if not isinstance(content, str):
            return None

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None

        aggregate = parsed.get("aggregate")
        if aggregate not in SUPPORTED_AGGREGATES and aggregate is not None:
            aggregate = None

        target_column = parsed.get("target_column")
        if target_column is not None and not isinstance(target_column, str):
            target_column = None

        filter_part = parsed.get("filter")
        if filter_part is not None:
            if not isinstance(filter_part, dict):
                filter_part = None
            elif not {"column", "op", "value"} <= set(filter_part.keys()):
                filter_part = None
            elif filter_part.get("op") not in {"<", "<=", ">", ">=", "==", "!="}:
                filter_part = None
            else:
                try:
                    filter_part["value"] = float(filter_part["value"])
                except (TypeError, ValueError):
                    filter_part = None

        return {
            "aggregate": aggregate,
            "target_column": target_column,
            "filter": filter_part,
        }

    except Exception:
        return None


def quickstats_intent(normalized_query: str) -> bool:
    tokens = set(_tokenize(normalized_query))
    multi_phrases = ["how many", "number of", "standard deviation"]

    if any(phrase in normalized_query for phrase in multi_phrases):
        return True

    single_word_keywords = SUMMARY_KEYWORDS + FILTER_KEYWORDS + [
        kw for kw in AGGREGATE_KEYWORDS if " " not in kw
    ]
    return any(keyword in tokens for keyword in single_word_keywords)


def summarize_intent(query: str) -> bool:
    tokens = set(_tokenize(query))
    return any(keyword in tokens for keyword in SUMMARY_KEYWORDS)


def filter_intent(query: str) -> bool:
    tokens = set(_tokenize(query))
    return any(keyword in tokens for keyword in FILTER_KEYWORDS)


def aggregate_intent(query: str) -> bool:
    normalized = query.lower()
    tokens = set(_tokenize(normalized))

    if "how many" in normalized or "number of" in normalized:
        return True

    single_word_agg = [kw for kw in AGGREGATE_KEYWORDS if " " not in kw]
    return any(keyword in tokens for keyword in single_word_agg)

#return the intent
def _route_intent(
    query: str,
    filter_spec: Optional[dict[str, Any]],
    aggregate_op: Optional[str],
) -> str:
    has_filter = bool(filter_spec) or filter_intent(query)
    has_aggregate = aggregate_intent(query) or aggregate_op in SUPPORTED_AGGREGATES

    if has_filter and has_aggregate:
        return "mixed"
    if has_filter:
        return "filter"
    if has_aggregate:
        return "aggregate"
    return "unknown"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

#map operation keyword to the pandas function and return the function
def _parse_aggregate(normalized_query: str) -> Optional[str]:
    for keyword, agg_func in AGGREGATE_DISPATCH.items():
        if keyword in normalized_query:
            return agg_func
    return None



def _parse_filter(normalized_query: str) -> Optional[dict[str, Any]]:
    #regex patterns for text patterns
    patterns = [
        re.compile(
            r"\b([a-z_][a-z0-9_]*)\b\s*(?:is\s+)?"
            r"(less|greater|more|fewer|below|above|under|over|equal(?:s| to)?)\s*"
            r"(?:than|to)?\s*(-?\d+(?:\.\d+)?)\b"
        ),
        re.compile(
            r"\b([a-z_][a-z0-9_]*)\b\s*(<=|>=|!=|=|==|<|>)\s*(-?\d+(?:\.\d+)?)\b"
        ),
    ]

    for pattern in patterns:
        #looks for first match in query
        match = pattern.search(normalized_query)
        if not match:
            continue

        column = match.group(1)
        comparator_token = match.group(2)
        value_text = match.group(3)
        op = OPERATOR_DISPATCH.get(comparator_token.lower())

        if op is None:
            continue

        return {"column": column, "op": op, "value": float(value_text)}

    return None

#load the dataframes from tabular files and create a copy for safe handling
def _collect_dataframes(tabular_files: List[Any]) -> List[pd.DataFrame]:
    dataframes: List[pd.DataFrame] = []

    for entry in tabular_files or []:
        if isinstance(entry, dict):
            for df in entry.get("dataframes") or []:
                if isinstance(df, pd.DataFrame):
                    copied = df.copy()
                    copied.attrs["source_name"] = entry.get("filename", "table")
                    dataframes.append(copied)
        elif isinstance(entry, pd.DataFrame):
            copied = entry.copy()
            copied.attrs["source_name"] = "dataframe"
            dataframes.append(copied)

    return dataframes

#tries converting each column’s values to numbers, not relying on pandas data type only
def _numeric_columns(df: pd.DataFrame) -> List[Any]:
    numeric_cols: List[Any] = []

    for col in df.columns:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                numeric_cols.append(col)
        except Exception:
            continue

    return numeric_cols

#filter df and apply operations, comparators, etc
def _apply_filter(df: pd.DataFrame, filter_spec: dict[str, Any]) -> pd.DataFrame:
    column = filter_spec["column"]
    if column not in df.columns:
        return None

    try:
        series = pd.to_numeric(df[column], errors="coerce")
    except Exception:
        return None

    op = filter_spec["op"]
    value = filter_spec["value"]
    comparator = FILTER_OPERATORS.get(op)

    if comparator is None:
        return None

    try:
        mask = comparator(series, value).fillna(False)
    except Exception:
        return None

    return df.loc[mask]

#perform aggregate operation 
def _aggregate_handler(
    df: pd.DataFrame,
    aggregate_op: Optional[str],
    label: str,
    target_column: Optional[str] = None,
) -> str:
    if aggregate_op == "count":
        return f"Count in {label}: {len(df)}"

    if aggregate_op is None:
        return f"No aggregate operation could be detected for {label}."

    if target_column is None:
        return f"No suitable numeric column found in {label} for the requested aggregate."

    if target_column not in df.columns:
        return f"Column '{target_column}' was not found in {label}."

    try:
        series = pd.to_numeric(df[target_column], errors="coerce").dropna()
    except Exception:
        return f"Unable to compute the requested aggregate for {label}."

    if series.empty:
        return f"No numeric values were available in {label} for the requested aggregate."
    #use pandas agg function
    try:
        value = series.agg(aggregate_op)
    except Exception:
        return f"Unsupported aggregate operation for {label}."

    display_value = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{aggregate_op.title()} of {target_column} in {label}: {display_value}"


#apply filter and then perform aggregate operation
def _mixed_handler(
    df: pd.DataFrame,
    filter_spec: Optional[dict[str, Any]],
    aggregate_op: Optional[str],
    label: str,
    target_column: Optional[str] = None,
) -> str:
    filtered_df = df.copy()
    if filter_spec:
        filtered_df = _apply_filter(filtered_df, filter_spec)

    if filtered_df is None:
        return f"Column '{filter_spec['column']}' was not found in {label}."

    if filtered_df.empty:
        return f"No rows matched the condition in {label}."

    return _aggregate_handler(
        df=filtered_df,
        aggregate_op=aggregate_op,
        label=label,
        target_column=target_column,
    )


def _select_target_column(
    normalized_query: str,
    df: pd.DataFrame,
    filter_spec: Optional[dict[str, Any]],
) -> Optional[str]:
    filter_column = None

    if filter_spec and filter_spec.get("column") in df.columns:
        filter_column = filter_spec["column"]

    for column in df.columns:
        if isinstance(column, str):
            column_name = column.lower()
            if column_name in normalized_query and column_name != (filter_column or ""):
                return column

    numeric_columns = _numeric_columns(df)
    if filter_column is not None:
        numeric_columns = [col for col in numeric_columns if col != filter_column]

    if numeric_columns:
        return numeric_columns[0]

    return None