from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

CACHE_FOLDER_PATH ="cache/"
DATA_PATH = "data/"
DB_PATH = "vectorDB/"
NUM_RETRIEVAL = 4

#GPU changes:
#Embeddings:
#   change device = 'cuda' for gpu
#CTRANSFORMERS_CONFIG
#   'gpu_layers': 50

EMBEDDINGS  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}, encode_kwargs={"normalize_embeddings": False}, cache_folder= CACHE_FOLDER_PATH)

CTRANSFORMERS_CONFIG = {'max_new_tokens': 256, 'gpu_layers': 0, 'temperature': 0.01, 'top_p': 0.4, 'repetition_penalty': 1.1, 'context_length': 3500}

LLM = CTransformers(
        model = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama",
        config = CTRANSFORMERS_CONFIG
    )

#prompt = hub.pull("rlm/rag-prompt")
#"You are an assistant for question-answering tasks. 
#Use the following pieces of retrieved context to answer the question. 
#If you don't know the answer, just say that you don't know. 
#Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:
CUSTOM_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
                        If you don't know the answer, just say that you don't know. 
                        Keep the answer short and concise. 
                        Add the original sources from the given Context at the end of your answer.
                        \nThe only Question for you is: {question} \nContext: {context} \nThe Answer:"""