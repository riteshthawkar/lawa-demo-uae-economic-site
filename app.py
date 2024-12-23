import nltk
nltk.download('punkt_tab')

import os
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.chat_models import ChatPerplexity
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import PromptTemplate
import re

# Load environment variables
load_dotenv(".env")
USER_AGENT = os.getenv("USER_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("huggingface_api_key")
SESSION_ID_DEFAULT = "abc123"

# Set environment variables
os.environ['USER_AGENT'] = USER_AGENT
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# Initialize FastAPI app and CORS
app = FastAPI()
origins = ["*"]  # Adjust as needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Function to initialize Pinecone connection
def initialize_pinecone(index_name: str):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

##################################################
##          Change down here
##################################################

# Initialize Pinecone index and BM25 encoder
pinecone_index = initialize_pinecone("uae-department-of-economics-site-data")
bm25 = BM25Encoder().load("./bm25_uae_department_of_economics_data.json")

##################################################
##################################################

# Initialize models and retriever
embed_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-large-en-v1.5", model_kwargs={"trust_remote_code":True})
retriever = PineconeHybridSearchRetriever(
    embeddings=embed_model, 
    sparse_encoder=bm25, 
    index=pinecone_index, 
    top_k=10, 
    alpha=0.5,
)


llm = ChatPerplexity(temperature=0, pplx_api_key=GROQ_API_KEY, model="llama-3.1-sonar-large-128k-chat", max_tokens=512, max_retries=2)


# Contextualization prompt and retriever
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA system prompt and chain
qa_system_prompt = """ You are a highly skilled information retrieval assistant. Use the following context to answer questions effectively. 
If you don't know the answer, simply state that you don't know.
YOUR ANSWER SHOULD BE IN '{language}' LANGUAGE. 
When responding to queries, follow these guidelines:
1. Provide Clear Answers: 
   - You have to answer in that language based on the given language of the answer. If it is English, answer it in English; if it is Arabic, you should answer it in Arabic.
   - Ensure the response directly addresses the query with accurate and relevant information.
   - Do not give long answers. Provide detailed but concise responses.
   
2. Formatting for Readability: 
   - Provide the entire response in proper markdown format.
   - Use structured Markdown elements such as headings, subheadings, lists, tables, and links.
   - Use emphasis on headings, important texts, and phrases.
   
3. Proper References:
   - Always use inline citations with embedded source URLs wherever needed to cite the sources. 
   - LIST OUT SOURCES URLs TO USERS TO REFER IN THE 'References' SECTION AT THE END RESPONSE.
   - The references list should be ordered list with proper order of sources given.
   
FOLLOW ALL THE GIVEN INSTRUCTIONS, FAILURE TO DO SO WILL RESULT IN THE TERMINATION OF THE CHAT.
== CONTEXT ==
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content} \n\n")
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)

# Retrieval and Generative (RAG) Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat message history storage
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Conversational RAG chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    language_message_key="language",
    output_messages_key="answer",
)


# WebSocket endpoint with streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"Client connected: {websocket.client}")
    session_id = None
    try:
        while True:
            data = await websocket.receive_json()
            question = data.get('question')
            language = data.get('language')
            if "en" in language:
                language = "English"
            else:
                language = "Arabic"
            session_id = data.get('session_id', SESSION_ID_DEFAULT)
            # Process the question
            try:
                # Define an async generator for streaming
                async def stream_response():
                    complete_response = ""
                    async for chunk in conversational_rag_chain.astream(
                        {"input": question, 'language': language},
                        config={"configurable": {"session_id": session_id}}
                    ):
                        # Send each chunk to the client
                        if "answer" in chunk:
                            complete_response += chunk['answer']
                            await websocket.send_json({'response': chunk['answer']})

                await stream_response()
            except Exception as e:
                print(f"Error during message handling: {e}")
                await websocket.send_json({'response': "Something went wrong, Please try again." + str(e)})
    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
        if session_id:
            store.pop(session_id, None)

# Home route
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})