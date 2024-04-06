import chainlit as cl
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import os
from operator import itemgetter
from data_handler.local_llm import LocalLLM
    
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
    NUM_RETRIEVAL = 5

    # Adds Retrieval prompt at the start of the query as necessiated by the EMBEDDING LLM.
    def get_retrieval_query(query):
        return __class__.RETRIEVAL_PROMPT + query

    def get_retrieved_context(docs):
        return "\n\n".join(f"{doc.page_content}" for doc in docs)

    def get_retrieved_sources(docs):
        sources_list = []
        for doc in docs:
            sources_list.append(doc.metadata)
        
        unique_sources_list = [dict(t) for t in {tuple(s.items()) for s in sources_list}]
        sorted_sources_list = sorted(unique_sources_list, key=lambda x: (x['source'], x['page']))
        
        # clubbing all the source PDFs pages into one entry to remove duplication
        clubbed_pages_dict = {}
        for entry in sorted_sources_list:
            if entry['source'] not in clubbed_pages_dict:
                clubbed_pages_dict[entry['source']] = str(entry['page'])
            else:
                clubbed_pages_dict[entry['source']]+=", "+ str(entry['page'])

        sources_string = ""
        for source, pages in clubbed_pages_dict.items():
            sources_string += source.split("/")[-1]
            sources_string +=", page(s):" + pages
            sources_string +="\n"
        
        return sources_string

    def merge_response_and_sources(response_sources):
        print("Printing llm_response and sources")
        print(response_sources)
        if not response_sources.get('sources'):
            return """Sorry, I don't have sufficient information on this topic. 
                Try adding more texts regarding this topic in my documents folder 
                for me to get helpful context about this topic."""
        else:
            return f"{response_sources.get('llm_response')}\n\nSources:\n{response_sources.get('sources')}"


    # Function to persist the processed texts to a vector database
    def persist_texts(texts):
        db = FAISS.from_documents(texts, __class__.EMBEDDINGS)
        db.save_local(__class__.DB_PATH)

    # Creates a runnable chain for the language model by setting up a FAISS database retriever,
    # and a prompt template, and then chaining them together with the LLM
    @cl.cache
    def get_retrieval_chain():
        db = FAISS.load_local(__class__.DB_PATH, __class__.EMBEDDINGS, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': __class__.NUM_RETRIEVAL, 
                                                   'search_type': "similarity_score_threshold", 'score_threshold': 0.8})
        
        retrieval_chain = RunnableParallel(
            {"question": RunnablePassthrough() | __class__.get_retrieval_query, "retrieved_docs": retriever }
        )
        
        llm_response_and_source_chain = (retrieval_chain
                                         | {
                                              "question" : itemgetter("question"),
                                              "context" : itemgetter("retrieved_docs") | RunnableLambda(__class__.get_retrieved_context),
                                              "sources" : itemgetter("retrieved_docs") | RunnableLambda(__class__.get_retrieved_sources),                                              
                                          }   
                                          | {
                                            "llm_response": LocalLLM.get_LLM_chain,
                                            "sources" : itemgetter("sources")
                                            }
                                         | __class__.merge_response_and_sources
                                         )
                                  
        return llm_response_and_source_chain