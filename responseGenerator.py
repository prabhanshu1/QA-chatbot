import constants
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
import time


def getCustomPromptTemplate():
    prompt = PromptTemplate.from_template(constants.CUSTOM_PROMPT_TEMPLATE)
    return prompt

def format_docs(docs):
    return "\n\n".join(f"{doc.page_content} \n {doc.metadata}" for doc in docs)

def getRunnableChain():
    db = FAISS.load_local(constants.DB_PATH, constants.EMBEDDINGS)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': constants.NUM_RETRIEVAL, 'score_threshold': 0.8})
    
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    
    chain = setup_and_retrieval | getCustomPromptTemplate() | constants.LLM | StrOutputParser()
    return chain


if __name__ == "__main__":
    t_start = time.perf_counter() #Timer start
    chain = getRunnableChain()
    print("Please enter your query.")
    user_input = str(input())
    response = chain.invoke(user_input)
    print(response)
    
    t_end = time.perf_counter()  #Timer ends
    print(f"Total time taken: {t_end - t_start:0.2f} seconds")



@cl.on_chat_start
async def on_chat_start():
    print("on_chat_start")
    cl.user_session.set("runnable", getRunnableChain())
    print("on_chat_start2")

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        # message.content + "Economic Survey 2022-23",
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    await msg.send()