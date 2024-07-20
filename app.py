import chainlit as cl
import os
os.environ["REPLICATE_API_TOKEN"] = "r8_Q4UuRhxj3U3u7GBsxbXNVqzJp9QQXgi1tHQsH"
print(os.environ.get("REPLICATE_API_TOKEN"))
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


llm = Replicate(
    model="meta/llama-2-70b-chat",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
)

def use_my_model():
    prompt = input()
    output = llm(prompt)
    
    for i in range(1, len(output), 1):
        print(output[i] , end = "")    
        
    return output

@cl.on_chat_start
async def on_chat_start():
    #load model
    model = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.''',
            ),
            ("human", "{question}"),
        ]
    )
    #the initial prompt

    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content = "")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # Send a response back to the user
        await msg.send()
