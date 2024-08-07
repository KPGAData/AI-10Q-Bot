import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages.ai import AIMessageChunk
from langchain.schema.runnable.config import RunnableConfig
from langchain.globals import set_debug
from llama_parse import LlamaParse

set_debug(False)

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN. 
parser = LlamaParse(result_type='markdown', verbose=True, language='en')

pdf_documents = parser.load_data('./data/10Q-AirBnB.pdf')

# This class is for converting a python dict into an python object
class DataObj:
    def __init__(self, data):
       for key, value in data.items():
           setattr(self, key, value)

# LlamaParse produces documents that don't have `page_content` attribute expected by Recursive Splitter`
document_dicts = [{"page_content": d.text, "metadata": {}} for d in pdf_documents]
documents = [DataObj(d) for d in document_dicts]
# print(documents[0].page_content)

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

### 3. LOAD HUGGINGFACE EMBEDDINGS
# hf_embeddings = HuggingFaceEndpointEmbeddings(
#     model=HF_EMBED_ENDPOINT,
#     task="feature-extraction",
#     huggingfacehub_api_token=HF_TOKEN,
# )
hf_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

FAISS_VECTOR_STORE = "FAISS"
QDRANT_VECTOR_STORE = "QDRANT"

VECTOR_STORE = QDRANT_VECTOR_STORE

hf_retriever = ""

if VECTOR_STORE == FAISS_VECTOR_STORE:
    DATA_DIR = "./data"
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")
    VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")

    FAISS_MAX_FETCH_SIZE = 5
    FAISS_MAX_BATCH_SIZE = 32
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR,
            hf_embeddings, 
            allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
        )
        print("Loaded Vectorstore at " + VECTOR_STORE_DIR)
    else:
        print("Indexing Files")
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        ### 4. INDEX FILES
        ### NOTE: REMEMBER TO BATCH THE DOCUMENTS WITH MAXIMUM BATCH SIZE = 32
        for i in range(0, len(split_documents), FAISS_MAX_BATCH_SIZE):
            if i==0:
                vectorstore = FAISS.from_documents(split_documents[i:i+FAISS_MAX_BATCH_SIZE], hf_embeddings)
                continue
            vectorstore.add_documents(split_documents[i:i+FAISS_MAX_BATCH_SIZE])
        vectorstore.save_local(VECTOR_STORE_DIR)

    # hf_retriever = vectorstore.as_retriever(search_kwargs={"k": FAISS_MAX_FETCH_SIZE, "fetch_k": FAISS_MAX_FETCH_SIZE})
    hf_retriever = vectorstore.as_retriever()
else:
    QDRANT_MAX_FETCH_SIZE = 2
    QDRANT_MAX_BATCH_SIZE = 32

    vectorstore = ""
    for i in range(0, len(split_documents), QDRANT_MAX_BATCH_SIZE):
        if i==0:
            vectorstore = Qdrant.from_documents(
                split_documents[i:i+QDRANT_MAX_BATCH_SIZE], 
                hf_embeddings, 
                location=":memory:",
                collection_name="10Q_ABNB"
            )
            continue
        vectorstore.add_documents(split_documents[i:i+QDRANT_MAX_BATCH_SIZE])
    
    # hf_retriever = CustomQdrantRetriever(vectorstore=vectorstore, top_k=QDRANT_MAX_FETCH_SIZE)

    # hf_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    hf_retriever = vectorstore.as_retriever()

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""

### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
# hf_llm = HuggingFaceEndpoint(
#     endpoint_url=HF_LLM_ENDPOINT,
#     max_new_tokens=64,
#     top_k=10,
#     top_p=0.95,
#     temperature=0.3,
#     repetition_penalty=1.15,
#     huggingfacehub_api_token=HF_TOKEN,
# )

hf_llm = ChatOpenAI(model="gpt-4o")

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "AirBnB 10Q agent"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt | hf_llm
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        if (isinstance(chunk, AIMessageChunk)):
            await msg.stream_token(chunk.content)
        else:
            await msg.stream_token(chunk)

    await msg.send()