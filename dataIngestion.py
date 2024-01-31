import constants
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import time


def processPDFData():
    loader = DirectoryLoader(constants.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader) ## try single mode 
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50) # add custom separator \n \n\n . Also Tokeniser
    texts = text_splitter.split_documents(documents)
    return texts


def persistTexts(texts):
    db = FAISS.from_documents(texts, constants.EMBEDDINGS)
    db.save_local(constants.DB_PATH)



if __name__ == "__main__":
    t_start = time.perf_counter() #Timer start
    texts = processPDFData()
    t_process = time.perf_counter()  #Timer ends
    print(f"Total time taken: {t_process - t_start:0.2f} seconds")
    
    ## Persisting the processed texts to vectorDB
    persistTexts(texts)
    t_persist = time.perf_counter()  #Timer ends
    print(f"Total time taken: {t_persist - t_process:0.2f} seconds")