import os
import shutil
import gradio as gr
#from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage


#load_dotenv()

# ---------------- CONFIG ----------------
PDF_PATH = "./docs/movies_trivia.pdf"
CHROMA_DIR = "./my_chroma_db"
EMBED_MODEL = "embeddinggemma:latest"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 20
# ---------------------------------------

# clean old vector DB
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

# load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(docs)

# embeddings + vector DB
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vector_db = Chroma(
    collection_name="movie_trivia",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
vector_db.add_documents(chunks)

# LLM
llm = ChatOllama(model=LLM_MODEL)


prompt = ChatPromptTemplate.from_template(
        """
You are a helpful movie trivia assistant.
Use the context and chat history to answer questions.

IMPORTANT RULES:
1. If the question uses pronouns like "it", "that movie", "this film" - refer to the previous conversation to identify which movie.
2. Use the chat history to understand what the user is talking about.
3. provide concise and accurate answers.

Chat History:
{history}

Relevant Information from Knowledge Base:
{context}

Current Question:
{question}

Answer:
"""
)

# runnable
runnable = RunnableLambda(
    lambda inputs: llm.invoke(prompt.invoke(inputs))
)

# memory store
history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

# chain with memory
chain = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# RAG function
def rag(question, session_id="user1"):
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    history = get_session_history(session_id).messages
    
    history_text = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"
    
   
    search_query = question
    
    
    pronouns = ["it", "this", "that", "its", "their", "they"]
    if any(pronoun in question.lower().split() for pronoun in pronouns):
        if len(history) >= 2:
            last_exchange = " ".join([m.content for m in history[-4:]])
            search_query = last_exchange + " " + question
    
    
    docs = retriever.invoke(search_query)

    context = "\n\n".join(d.page_content for d in docs)

    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "history":history_text
        },
        config={"configurable": {"session_id": session_id}}
    )
    return response
def chat_interface(message, history):
    """Gradio chat function"""
    session_id = "gradio_user"
    
    try:
        response = rag(message, session_id)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Add title
    gr.Markdown("# 🎬 MovieBot - Your Movie Trivia Expert")
    chatbot = gr.Chatbot(height=400) 
    msg = gr.Textbox(
        placeholder="Type your question here...",
        show_label=False
    )
    with gr.Row():
        submit_btn = gr.Button("🚀 Submit", variant="primary", size="lg")
        clear_btn = gr.Button("🗑️ Clear History", variant="primary", size="lg")
    def show_greeting():
        return [{"role": "assistant", "content": "Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!"}]
    def respond(message, history):
        if not message or not message.strip():
            return history, ""
        bot_message = chat_interface(message, history)
        # Add user message
        history.append({"role": "user", "content": message})
        # Add bot message
        history.append({"role": "assistant", "content": bot_message})
        return history, ""
    
    def clear_all():
        session_id = "gradio_user"
        if session_id in history_store:
            history_store[session_id] = InMemoryChatMessageHistory()
        return []
    # Load greeting on page load
    demo.load(
        fn=show_greeting,
        outputs=chatbot
    )
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_all, None, chatbot)
 
print("\n" + "="*60)
print("🚀 Launching Gradio interface...")
print("="*60)
 
demo.launch(share=True)
 