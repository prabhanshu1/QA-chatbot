from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os

# Handles all the Database related executions
class DataManager():
    # path to store local DB created to persists processed documents.
    DB_PATH = os.path.dirname(__file__) + "/vectorDB/"

    #Path to store Embedding models.
    CACHE_FOLDER_PATH = os.path.dirname(__file__) + "/cache/"

    ## Embeddings setting:
    #   change device = 'cuda' for gpu; or 'cpu' for using only CPU.
    EMBEDDINGS_BASE_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDINGS  = HuggingFaceBgeEmbeddings(model_name=EMBEDDINGS_BASE_MODEL, model_kwargs={'device': 'cuda'}, 
                                           encode_kwargs={"normalize_embeddings": True}, cache_folder= CACHE_FOLDER_PATH)
    

    ## Retrieval settings:
    #   prompt suggested by EMBEDDINGS_BASE_MODEL: https://huggingface.co/BAAI/bge-large-en-v1.5
    RETRIEVAL_PROMPT = "Represent this sentence for searching relevant passages: "
    NUM_RETRIEVAL = 4

    # Adds Retrieval prompt at the start of the query as necessiated by the EMBEDDING LLM.
    def get_retrieval_query(query):
        return __class__.RETRIEVAL_PROMPT + query

    def format_retrieved_docs(docs):
        return "\n\n".join(f"{doc.page_content} \n {doc.metadata}" for doc in docs)


    # Function to persist the processed texts to a vector database
    def persist_texts(texts):
        db = FAISS.from_documents(texts, __class__.EMBEDDINGS)
        db.save_local(__class__.DB_PATH)

    
    def get_runnableChain_for_retrieval():
        db = FAISS.load_local(__class__.DB_PATH, __class__.EMBEDDINGS, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': __class__.NUM_RETRIEVAL, 'score_threshold': 0.8})
        
        setup_and_retrieval_chain = RunnableParallel(
            {"question": RunnablePassthrough() | __class__.get_retrieval_query, "context": retriever | __class__.format_retrieved_docs }
        )
        return setup_and_retrieval_chain