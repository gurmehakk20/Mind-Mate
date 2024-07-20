from langchain_community.llms import CTransformers
from chainlit as cl

# Function definitions
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def create_chatbot():
    llm = load_llm()
    return llm

def chat(chatbot, message):
    response = chatbot.run(message, output_prefix="FINAL ANSWER: ")
    return response

# Chainlit
@cl.on_chat_start
async def start():
    chatbot = create_chatbot()
    msg = cl.Message(content="Hi, welcome to MindMate!")
    await msg.send()

@cl.on_message()
async def on_message(message):
    response = chat(cl.user_session.get("chatbot"), message)
    await cl.Message(content=response).send()

cl.run()
