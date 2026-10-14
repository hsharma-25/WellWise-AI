from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from split_docs import chunks
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="data/chroma_db",
    collection_name="offset-well-reports"
)