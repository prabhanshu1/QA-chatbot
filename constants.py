from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

DATA_PATH = "data/"
DB_PATH = "vectorDB/"
NUM_RETRIEVAL = 1


EMBEDDINGS  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})


LLM = CTransformers(
        model = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama",
        max_new_tokens = 100,
        temperature = 0
    )

#prompt = hub.pull("rlm/rag-prompt")
#"You are an assistant for question-answering tasks. 
#Use the following pieces of retrieved context to answer the question. 
#If you don't know the answer, just say that you don't know. 
#Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:
CUSTOM_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""