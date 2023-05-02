# Import required libraries
import streamlit as st
import os
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import shutil
from streamlit_chat import message

message_history= []
# Set page title
st.set_page_config(page_title="JKAssist",page_icon=":robot_face:", layout="wide")
# st.set_page_config()
os.environ["OPENAI_API_KEY"] = '<Your API Key>'

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Define function to construct index
def construct_index(directory_path):
    # Set parameters
    max_input_size = 4096
    num_outputs = 2048
    max_chunk_overlap = 20
    chunk_size_limit = 2000

    # Initialize prompt helper and LLM predictor
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    # Load documents and create index
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Save index to disk
    index.save_to_disk('index.json')

    return index


# Define function to generate chatbot response
def chatbot(input_text):
    st.session_state['messages'].append({"role": "user", "content": input_text})
    # Load index and generate response
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    st.session_state['messages'].append({"role": "assistant", "content": response})
    # chat_history.append({"input": input_text, "response": response.response})

    return response.response



# Define function to reset index
def reset_index():
    shutil.rmtree('Data')
    
    os.remove('index.json')
    
    print("Resetting Index")
    st.sidebar.warning("Index reset. Please rebuild the index.")



    # Create Data folder if it doesn't exist
if not os.path.exists('Data'):
        os.makedirs('Data')

# Add logo to sidebar
from PIL import Image
logo = Image.open("logo.png")
st.sidebar.image(logo, use_column_width=True)

# Create sidebar
st.sidebar.title("Chatbot Configuration")
uploaded_file = st.sidebar.file_uploader("Upload a file")
if uploaded_file is not None:
    with open(os.path.join('Data', 'docs' + os.path.splitext(uploaded_file.name)[1]), "wb") as f:
        f.write(uploaded_file.getvalue())
    # Construct the index using the uploaded file
if st.sidebar.button("Construct Index"):
    # Show progress bar
    progress_bar = st.sidebar.progress(0)
    print("Constructing Index")
    index = construct_index('Data')
    
    # Update progress bar
    progress_bar.progress(100)
    st.sidebar.write("Index constructed successfully!")
    index_built = True




# Create main page
# st.title("Contextual LLM AI Chatbot")
st.markdown("<h1 style='text-align: center; color: black;'>Contextual LLM AI Chatbot</h1>", unsafe_allow_html=True)

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        input_text = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and input_text:
        output = chatbot(input_text)
        st.session_state['past'].append(input_text)
        st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',avatar_style='initials', seed = "U")
            message(st.session_state["generated"][i], key=str(i), avatar_style='initials', seed = "JA")


# Add "Reset Index" button to sidebar
if st.sidebar.button("Reset Index"):
    reset_index()
    index_built = False

# Check if index.json file already exists
if os.path.exists("index.json"):
    st.sidebar.warning("An index already exists.")

clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.markdown(
    """
    <style>
        .stSideBar {
            background-color: #f5f5f5;
        }
        .my-button {
            background-color: green;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
