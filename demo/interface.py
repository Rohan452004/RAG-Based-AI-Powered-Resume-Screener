import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import google.generativeai as genai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

print(DATA_PATH)
print(FAISS_PATH)

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your Google Gemini API key. 🔑 
  2. Type in a job description query. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Rohan452004/RAG-Based-AI-Powered-Resume-Screener/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Rohan452004/RAG-Based-AI-Powered-Resume-Screener/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the Gemini model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by Google's Gemini**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

# about_message = """
#   # About

#   This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
#   The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

#   The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

#   If you are interested, please don't hesitate to give me a star. ⭐
# """


st.set_page_config(page_title="Resume Screening with Gemini")
st.title("Resume Screening with Gemini")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  if st.session_state.uploaded_file != None:
    try:  
      df_load = pd.read_csv(st.session_state.uploaded_file)
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your csv file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          st.session_state.retriever = SelfQueryRetriever(vectordb, st.session_state.df)
  else:
    st.session_state.df = pd.read_csv(DATA_PATH)
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)


def check_gemini_api_key(api_key: str):
  if not api_key or api_key.strip() == "":
    return False
  # Basic validation - just check if the API key format is correct
  # The actual model validation will happen when we try to use it
  if not api_key.startswith("AIza"):
    return False
  # Try to validate by checking if we can create a client
  try:
    # Just verify the key format and that we can initialize
    # We'll do actual model validation when the user tries to use it
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    # Try to list models as a basic check
    try:
      list(genai.list_models())
      return True
    except:
      # If listing fails, the key might still be valid, just return True
      # The actual error will show when trying to use a model
      return True
  except Exception as e:
    return False
  
  
def check_model_name(model_name: str, api_key: str):
  if not api_key or api_key.strip() == "":
    return False
  # Just check if a model name is provided - actual validation happens when used
  # Common Gemini model name patterns
  if model_name and len(model_name.strip()) > 0:
    return True
  return False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  st.text_input("Google Gemini API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG"], placeholder="Generic RAG", key="rag_selection")
  st.selectbox("Gemini Model", 
               ["gemini-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.0-pro"],
               index=0,
               key="model_selection")
  st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file", on_change=upload_file)
  st.button("Clear conversation", on_click=clear_message)

  st.divider()
  st.markdown(info_message)

  # st.divider()
  # st.markdown(about_message)
  # st.markdown("Made by [Hungreeee](https://github.com/Hungreeee)")


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


if not st.session_state.api_key:
  st.info("Please add your Google Gemini API key to continue. Learn more about [API keys](https://ai.google.dev/gemini-api/docs/api-key).")
  st.stop()

if not check_gemini_api_key(st.session_state.api_key):
  st.error("The API key format is invalid. Please ensure:")
  st.error("1. Your API key is correct (should start with 'AIza...')")
  st.error("2. The API key has no extra spaces or characters")
  st.error("Learn more about [API keys](https://ai.google.dev/gemini-api/docs/api-key)")
  st.stop()

if not check_model_name(st.session_state.model_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [Gemini models](https://ai.google.dev/gemini-api/docs/models/gemini).")
  st.stop()


retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=st.session_state.api_key,
  model=st.session_state.model_selection,
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))