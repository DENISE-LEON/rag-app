#routes mode to pipeline
#helper methods to determine the best mode based on user intent 
from backend.core.aggregates import quickstats_intent


#function to determine the best mode based on user intent and file types
def determine_best_mode(query, has_tabular, has_text):
    #data intents
    if has_tabular:
        if quickstats_intent(query):
            return "quickstats", "you have tabular data and quickstats intent"
        elif has_text:
            return "analysis","you have tabular data and text files"
        else:
            return "analysis", "you have tabular data but no quickstats intent"
    #rag intents
    elif has_text and not has_tabular:
        return "rag", "you have text files only"
    else:
        # Default to RAG if no files are present
        return "rag", "no files found"