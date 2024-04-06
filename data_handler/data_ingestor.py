from data_handler.data_manager import DataManager
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from operator import itemgetter
import re
import time

class DataIngestor():

    # All supported Documents including PDFs within this path would be processed
    DOCUMENT_DIRECTORY = os.path.dirname(__file__) + "/documents/"
    
    # Set of all the DirectoryLoader object - each for different type of documents
    SUPPORTED_DOCUMENT_TYPES_AND_LOADERS = {
        DirectoryLoader(DOCUMENT_DIRECTORY, glob='*.pdf',loader_cls=PyPDFLoader)
    }

    # Function to clean PDF documents by replacing tabs, removing hyphens at the end of lines,
    # replacing single newlines with spaces, and reducing multiple newlines to two newlines
    # cleaning PDF before ingestion to increase effectiveness of similarity search
    
    def document_cleaner(documents):
        for doc in documents:
            doc = doc.page_content.replace('\t', ' ')
            doc = re.sub(r"(\w+)-\n(\w+)", r"\1\2", doc) # combine hyphenated words that are split across lines
            doc = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", doc.strip()) # replace single newlines with spaces
            doc = re.sub(r"\n\s*\n", "\n\n", doc) #  reduce newline followed by spaces and newline to two newlines
        return documents
    
    
    def split_document_into_chunks(documents):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", "!", "?"],
                                                    chunk_size=700,
                                                    chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        return texts
    
    # Process all supported documents (e.g., PDFs) stored in the DOCUMENT_DIRECTORY.
    # It loads the documents as text, cleans it and split into chunks for storing into vectorDB.
    
    def process_documents():
        docs = []

        for loader in __class__.SUPPORTED_DOCUMENT_TYPES_AND_LOADERS:
            docs.extend(loader.load())
        
        cleaned_docs = __class__.document_cleaner(docs)
        
        text_chunks = __class__.split_document_into_chunks(cleaned_docs)

        return text_chunks



    # Main function that processes and persists the texts from PDF files
    # Has to be run to process new PDF files;
    # In case of deletion or updation of the PDFs, first delete the vectorStore.
    def ingest_documents():
        t_start = time.perf_counter() #Timer for processing PDF
        
        #processes all the documents stored
        texts = DataIngestor.process_documents()
        
        t_process = time.perf_counter()
        print(f"Total time taken to process PDFs: {t_process - t_start:0.2f} seconds")
        
        ## Persisting the processed texts to vectorDB
        DataManager.persist_texts(texts)
        t_persist = time.perf_counter()  
        print(f"Total time taken to process and persist all the PDFs: {t_persist - t_process:0.2f} seconds")