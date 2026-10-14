from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from doc_loader import docs

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

chunks = splitter.split_documents(docs)