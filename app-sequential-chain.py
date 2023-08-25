import streamlit as st
from langchain.llms import OpenAI
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
    else:
        print("OPENAI_API_KEY is set")




def main():
    init()
    st.title("Youtube Script Generator with LangChain 🦜🔗")

    prompt = st.text_input("Please enter your prompt here")

    ##Prompt template
    title_template = PromptTemplate(input_variables=['topic'],
                                    template="Write me a Youtube video title about {topic}")

    script_template = PromptTemplate(input_variables=['title'],
                                     template="Write a youtube script base on the TITLE {title}.")

    ##memory
    memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    ##LLM
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)

    sequential_chain = SequentialChain(chains=[title_chain, script_chain],
                                       input_variables=['topic'],
                                       output_variables=['title', 'script'],
                                       verbose=True)

    if prompt:
        response = sequential_chain({'topic': prompt})
        st.write(response['title'])
        st.write(response['script'])

        with st.expander("Message History:"):
            st.info(memory.buffer)


if __name__ == '__main__':
    main()
