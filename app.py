import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
# Load the .env file

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') 
groq_api_key = os.getenv('GROQ_API_KEY') 

# llm
llm = ChatGroq(model='llama3-8b-8192',groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
provide the important summary of the following content
Content:{text}

"""
# Prompt
prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

# Streamlit App
st.set_page_config(page_title="LangChain Summarization",page_icon="🦜")
st.title("Summarize text from YouTube or Website")
st.subheader("Summarize URL")
base_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize"):
    # Validate all the inputs
    if not base_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(base_url):
        st.error("Please provide the valid URL.It can be a YouTube video or Website URL ")

    else:
        try:
            with st.spinner("Processing..."):
                if "youtube.com" in base_url or "youtu.be" in base_url:
                    loader = YoutubeLoaderDL.from_youtube_url(youtube_url=base_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[base_url],ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
            
            docs = loader.load()

            # Chain
            chain = load_summarize_chain(llm,chain_type='stuff',prompt= prompt)
            summary=chain.run(docs)
            st.success(summary)
        
        except Exception as ex:
            st.exception(f"{ex}")
