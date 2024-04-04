import chainlit as cl
from data_handler.data_manager import DataManager
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable.config import RunnableConfig
from text_generation import TextGeneration
import time


# Creates a runnable chain for the language model by setting up a FAISS database retriever,
# and a prompt template, and then chaining them together with the LLM
@cl.cache
def get_runnable_chain():
    retrieval_chain = DataManager.get_runnableChain_for_retrieval()
    llm_chain = TextGeneration.get_LLM_chain()
    return (retrieval_chain | llm_chain)
    # db = FAISS.load_local(constants.DB_PATH, constants.EMBEDDINGS, allow_dangerous_deserialization=True)
    # retriever = db.as_retriever(search_kwargs={'k': constants.NUM_RETRIEVAL, 'score_threshold': 0.8})
    
    # setup_and_retrieval = RunnableParallel(
    #     {"question": RunnablePassthrough() | format_query, "context": retriever | format_docs }
    # )
    
    # chain = setup_and_retrieval | getCustomPromptTemplate() | constants.LLM | StrOutputParser()
    # return chain


# Function necessary to run the script from command line.
if __name__ == "__main__":
    t_start = time.perf_counter() #Timer start
    chain = get_runnable_chain()
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
    cl.user_session.set("runnable", get_runnable_chain())


# called when the user sends a message. 
# It retrieves the runnable chain from the user session object, and uses it to create response.
# The response are then streamed to the user.
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  #User specific langchain runnableChain.
    msg = cl.Message(content="") # will store the output response to tbe streamed as a reply
    
    async for chunk in runnable.astream(
        message.content, # will be passed to RunnablePassthrough() in the getRunnableChain() function to "question" variable
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]), #for tracing the execution
    ):
        await msg.stream_token(chunk)
    
    await msg.send()