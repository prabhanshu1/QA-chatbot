from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import CTransformers

CACHE_FOLDER_PATH ="cache/"

# All PDFs within this path would be processed
DATA_PATH = "data/"

DB_PATH = "vectorDB/"



#Embeddings setting:
#   change device = 'cuda' for gpu; or 'cpu' for using only CPU.
EMBEDDINGS_BASE_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDINGS  = HuggingFaceBgeEmbeddings(model_name=EMBEDDINGS_BASE_MODEL, model_kwargs={'device': 'cuda'}, encode_kwargs={"normalize_embeddings": True}, cache_folder= CACHE_FOLDER_PATH)


## Retrieval settings:
# prompt suggested by EMBEDDINGS_BASE_MODEL: https://huggingface.co/BAAI/bge-large-en-v1.5
RETRIEVAL_PROMPT = "Represent this sentence for searching relevant passages: "
NUM_RETRIEVAL = 4


#LLM settings and configuration
#   'gpu_layers': 50 - if GPU is available - more layer eqals more parallelisation, but cannot be too high i.e. more than its capacity.
#   'temperature': near zero - to force LLM to base its response on the context given.
CTRANSFORMERS_CONFIG = {'max_new_tokens': 512, 'gpu_layers': 50, 'temperature': 0.01, 'top_p': 0.4, 'repetition_penalty': 1.1, 'context_length': 8500}

LLM = CTransformers(
        model = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
        model_type="llama",
        config = CTRANSFORMERS_CONFIG
    )


# Prompts are LLM specific. So, if LLM is changed, then prompt may also has to be changed.
# This prompt is for LLM: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
CUSTOM_PROMPT_TEMPLATE = """
<|system|>
You are a helpful assistant for question-answering tasks based on given context only. If you don't know the answer, just say that you don't know. Keep the answer concise. You have to use the contexts given by the user to answer the question.
<|user|>
 The Context: {context}.\n The Question: {question}.
 <|assistant|>
 """