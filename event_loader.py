import os, json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document

load_dotenv()

with open("event.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for item in data:
    text = f"""
    행사명: {item.get('event_name')}
    일시: {item.get('date')} ({item.get('day')})
    장소: {item.get('location')}
    주최: {', '.join(item.get('hosts', []))}
    요약: {item.get('summary')}
    상세: {item.get('details')}
    카테고리: {item.get('category')}
    출처: {item.get('source')}
    """
    docs.append(Document(page_content=text.strip(), metadata={
        "date": item.get("date"),
        "location": item.get("location"),
        "category": item.get("category")
    }))

# 청킹
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splitted_docs = splitter.split_documents(docs)

# 임베딩
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_EMBED_KEY"),
    azure_endpoint=os.getenv("AZURE_EMBED_ENDPOINT"),
    api_version=os.getenv("AZURE_EMBED_API_VERSION"),
    azure_deployment=os.getenv("AZURE_EMBED_DEPLOYMENT")
)

# 벡터DB 저장
vectorstore = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    persist_directory="./paju_db"
)
vectorstore.persist()
print("저장완료")
