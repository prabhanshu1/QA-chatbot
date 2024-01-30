from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/"
DB_PATH = "vectorDB/"

EMBEDDINGS  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})