from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"
def create_vector_db() :
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls = PyPDFLoader)
    documents = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50) 
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs = {'device':'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)



# Function definitions
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}  # Fixed the typo here
    )
    return qa_chain
    
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response



@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Creating chunks for `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Write the file to local file system
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    with open(f"tmp/{file.name}", "wb") as f:
        f.write(file.content)

    pdf_loader = PyPDFLoader(file_path=f"tmp/{file.name}")

    # Split the text into chunks
    documents = pdf_loader.load_and_split(text_splitter=text_splitter)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(documents))]

    msg.content = f"Creating embeddings for `{file.name}`. . ."
    await msg.update()

    # Create a FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device':'cpu'})
    docsearch = await cl.make_async(FAISS.from_documents)(
        documents,
        embeddings
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

# Chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot....")
    await msg.send() 
    msg.content = "Hi, Welcome to MindMate. What is your query?"
    await msg.update() 
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()