from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path='data',
    glob='*.PDF',
    loader_cls=PyMuPDFLoader,
    loader_kwargs={"mode": "page"}
)

docs = loader.load()