import constants
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import time


def getCustomPromptTemplate():
    prompt = PromptTemplate.from_template(constants.CUSTOM_PROMPT_TEMPLATE)
    return prompt


def retrievalAndGeneration():
    db = FAISS.load_local(constants.DB_PATH, constants.EMBEDDINGS)
    retriever = db.as_retriever(search_kwargs={'k': constants.NUM_RETRIEVAL})
    
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    chain = setup_and_retrieval | getCustomPromptTemplate() | constants.LLM | StrOutputParser()
    return chain


if __name__ == "__main__":
    chain = retrievalAndGeneration()
    t_start = time.perf_counter() #Timer start
    
    print("Please enter your query.")
    user_input = str(input())
    response = chain.invoke(user_input)
    print(response)
    
    t_end = time.perf_counter()  #Timer ends
    print(f"Total time taken: {t_end - t_start:0.2f} seconds")
    