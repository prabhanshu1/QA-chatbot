import constants
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import time
import re

# Function to clean PDF documents by replacing tabs, removing hyphens at the end of lines,
# replacing single newlines with spaces, and reducing multiple newlines to two newlines
def cleanPDFDocuments(documents):
    for doc in documents:
        doc = doc.page_content.replace('\t', ' ')
        doc = re.sub(r"(\w+)-\n(\w+)", r"\1\2", doc) # remove hyphenated words that are split across lines
        doc = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", doc.strip()) # replace single newlines with spaces
        doc = re.sub(r"\n\s*\n", "\n\n", doc) #  reduce newline followed by spaces and newline to two newlines
    return documents

# Function to process PDF data by loading PDF files from a directory, cleaning the documents,
# and splitting the cleaned documents into chunks of text
def processPDFData():
    loader = DirectoryLoader(constants.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader) 
    documents = loader.load()
    cleaned_documents = cleanPDFDocuments(documents) #cleaning PDF before ingestion so that similarity search could be effective
    
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                                                    chunk_size=2000,
                                                    chunk_overlap=100)

    texts = text_splitter.split_documents(cleaned_documents)
    return texts

# Function to persist the processed texts to a vector database
def persistTexts(texts):
    db = FAISS.from_documents(texts, constants.EMBEDDINGS)
    db.save_local(constants.DB_PATH)


# Main function that processes and persists the texts from PDF files
# Has to be run to process new PDF files;
# In case of deletion or updation of the PDFs, first delete the vectorStore.
if __name__ == "__main__":
    t_start = time.perf_counter() #Timer for processing PDF
    texts = processPDFData()
    t_process = time.perf_counter()
    print(f"Total time taken to process PDFs: {t_process - t_start:0.2f} seconds")
    
    ## Persisting the processed texts to vectorDB
    persistTexts(texts)
    t_persist = time.perf_counter()  
    print(f"Total time taken to process and persist all the PDFs: {t_persist - t_process:0.2f} seconds")