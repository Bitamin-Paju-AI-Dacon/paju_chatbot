import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def retrieve_event_info(query: str, top_k: int = 2):
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_EMBED_KEY"),
        azure_endpoint=os.getenv("AZURE_EMBED_ENDPOINT"),
        api_version=os.getenv("AZURE_EMBED_API_VERSION"),
        azure_deployment=os.getenv("AZURE_EMBED_DEPLOYMENT")
    )

    db = Chroma(
        persist_directory="./paju_db",
        embedding_function=embeddings
    )

    results = db.similarity_search(query, k=top_k)
    return results
