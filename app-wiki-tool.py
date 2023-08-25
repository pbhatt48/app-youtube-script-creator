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

    script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                     template="Write a youtube script base on the TITLE {title} while leveraging this wikipedia research: {wikipedia_research}")

    ##memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    ##LLM
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    wiki = WikipediaAPIWrapper()

    if prompt:
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title=title, wikipedia_research=wiki_research)
        st.write(title)
        st.write(script)

        with st.expander("Title History"):
            st.info(title_memory.buffer)
        with st.expander("Script History:"):
            st.info(script_memory.buffer)
        with st.expander("Wikipedia History:"):
            st.info(wiki_research)

if __name__ =='__main__':
    main()