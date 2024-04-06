from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LocalLLM():
    
    #LLM settings and configuration
    #   'gpu_layers': 50 - if GPU is available - more layer eqals more parallelisation, but cannot be too high i.e. more than its capacity.
    #   'temperature': near zero - to force LLM to base its response on the context given.
    CTRANSFORMERS_CONFIG = {'max_new_tokens': 512, 'gpu_layers': 50, 'temperature': 0.0, 
                            'top_p': 0.4, 'context_length': 4096}

    LLM = CTransformers(
            model = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
            model_type="llama",
            config = CTRANSFORMERS_CONFIG
        )

    # Prompts are LLM specific. So, if LLM is changed, then prompt may also has to be changed.
    # This prompt is for LLM: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
    CUSTOM_PROMPT_TEMPLATE = """<|system|>
    You are a helpful assistant for question-answering tasks based on given context only. 
    Keep the answer concise and pointwise. You have to use the contexts given by the user to answer the question.
    If you don't know the answer, just say that you don't know. 
    <|user|>
    The Context: {context}.
    The Question: {question}.
    
    <|assistant|>
    """

    # Creates a custom prompt template in the required format for the LLM to generate output based on context
    def get_custom_prompt_template():
        prompt = PromptTemplate.from_template(__class__.CUSTOM_PROMPT_TEMPLATE)
        return prompt
    
    def get_LLM_chain(inp):
        chain = __class__.get_custom_prompt_template() | __class__.LLM | StrOutputParser()
        return chain