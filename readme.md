# QA-chatbot
Welcome to the QA-chatbot Repository.

It is a **question answering bot**. It is based on the **RAG technique** aimed at *reducing the hallucination* in the **Large Language Models(LLMs)** response by increasing its accuracy. This is achieved by basing the generation of the response based on specific set of data/context derieved from pdf documents that we have in the data_handler/documents folder.

## Features
- uses Retrieval Augmented Generation(RAG) techniqe to improve accuracy
- It also **cites the sources** of its information - for reference.
- Can try out different LLM models  by changing the model name in the local_llm.py

## How to use it: 🚀
You can check out this chatbot on [Google Colab](https://colab.research.google.com/drive/1q39WA6DOd9vZKsJ1WQ3ZgjA3ic3KPz2L?usp=sharing). It provides step by step guide to use this web app.


## Motivation for this project:
**To try out the new technologies** in the IT field, i.e. AI and particularly Generative AI.
I, like almost everyone else was following the developments in this field. But, I wanted to go further, to implement something and use LLM first hand. And, since **Retrieval Augmented Generation(RAG)** is gaining popularity among enterprises, I attempted to implement it.


## Software Design
1. Retrieval Augmented Generation(RAG) technique
    - improves the output of a large language model (LLM) by referencing an authoritative knowledge base outside of its training data sources before generating a response.
    - uses semantic similarity calculations to retrieve relevant documents from an external knowledge base, which reduces the risk of generating factually incorrect content.
    - Reduce the possibility of hallucinations: Also known as wrong guesses
2. The web app uses **MVC design pattern** to build the web-app.
    - Model: All the data related modules are in data_handler
    - View: Chainlit framework help in generating the HTML pages and styling them
    - Controller: app.py is the controller. The request are handled by the app.py and then uses other modules to process the requests.

## Tools/Frameworks used:
- Large Language Model - TinyLlama-1.1B-Chat-v1.0-GGUF from [HuggingFace](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- [Langchain Framework](https://www.langchain.com/) to handle the LLM related workflow smoothly and efficiently
- [Chainlit Package](https://docs.chainlit.io/get-started/overview) to build the website of the Web-app. 
    - manages user sessions.
    - It also has easy to use triggers like on_message, on_chat_start to initiate a workflow.
    - provides text-to-speech feature.
- [FAISS vectorDB](https://faiss.ai/index.html) - to store the embeddings of the texts and then retrieve contextual texts using similiarity search algorithm.
- [CTransformers Library](https://github.com/marella/ctransformers) - to load LLM  models from Hugging Face.
- [PyPDF Library](https://pypi.org/project/pypdf/) to load PDFs into texts.


## Further Improvements:
- Use better Embedding model and LLM Model
    - currently, The LLM model used has only 1.1 Bn parameters to run on a single GPU and still respond  with decent speed. In comparison the GPT-4 has 1.75 Trillion parameters.
- Further improving RAG - to improve accuracy
- Exception handling
- Sources as objects - separating code of processing sources information to separate module.