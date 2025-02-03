from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model = "deepseek-r1:8b"
texts_path = "./texts"
db_path = "./chroma_db"


# load the texts
loader = DirectoryLoader(texts_path, glob="**/*.txt")
texts = loader.load()

# split the text into chunks
text_splitter = CharacterTextSplitter()
all_splits = text_splitter.split_documents(texts)

print("Loaded texts:", len(texts))
print("Total number of splits:", len(all_splits))


# save splits into vector store
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    persist_directory=db_path,
)

### TEST
