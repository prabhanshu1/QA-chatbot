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

# Creates a custom prompt template in the required format for the LLM to generate output based on context
def getCustomPromptTemplate():
    prompt = PromptTemplate.from_template(constants.CUSTOM_PROMPT_TEMPLATE)
    return prompt
# Adds Retrieval prompt at the start of the query as necessiated by the EMBEDDING LLM.
def format_query(query):
    return constants.RETRIEVAL_PROMPT + query

def format_docs(docs):
    return "\n\n".join(f"{doc.page_content} \n {doc.metadata}" for doc in docs)

# Creates a runnable chain for the language model by setting up a FAISS database retriever,
# and a prompt template, and then chaining them together with the LLM
def getRunnableChain():
    db = FAISS.load_local(constants.DB_PATH, constants.EMBEDDINGS)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': constants.NUM_RETRIEVAL, 'score_threshold': 0.8})
    
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    
    chain = setup_and_retrieval | getCustomPromptTemplate() | constants.LLM | StrOutputParser()
    return chain

# Function necessary to run the script from command line.
if __name__ == "__main__":
    t_start = time.perf_counter() #Timer start
    chain = getRunnableChain()
    print("Please enter your query.")
    user_input = str(input())
    response = chain.invoke(user_input)
    print(response)
    
    t_end = time.perf_counter()  #Timer ends
    print(f"Total time taken: {t_end - t_start:0.2f} seconds")

## Chainlit module for Front-end and streaming response

# defines callback functions for the chainlit module when the user starts a chat session.
# it stores the user_session so that history of the chat could be used in subsequent interaction.
@cl.on_chat_start
async def on_chat_start():
    print("on_chat_start")
    cl.user_session.set("runnable", getRunnableChain())
    print("on_chat_start2")

# called when the user sends a message. 
# It retrieves the runnable chain from the user session object, and uses it to create response.
# The response are then streamed to the user.
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