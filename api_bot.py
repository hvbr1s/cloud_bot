import os
import uuid
import json
from typing import Callable, List, Union

from dotenv import load_dotenv
from flask import Flask, render_template, request, make_response, redirect, jsonify
from web3 import Web3
from eth_account.messages import encode_defunct

import pinecone
import openai

from langchain import LLMChain
from langchain.agents import initialize_agent, load_tools, ZeroShotAgent, AgentExecutor, Tool, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain, ConversationalRetrievalChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

import re



load_dotenv()
history = ChatMessageHistory()

env_vars = [
    'OPENAI_API_KEY',
    'ALCHEMY_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT',
]

os.environ.update({key: os.getenv(key) for key in env_vars})
os.environ['WEB3_PROVIDER'] = f"https://polygon-mumbai.g.alchemy.com/v2/{os.environ['ALCHEMY_API_KEY']}"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
ORGANIZATION = os.getenv("ORGANIZATION")
openai.api_key=os.environ['OPENAI_API_KEY']

# Initialize web3
web3 = Web3(Web3.HTTPProvider(os.environ['WEB3_PROVIDER']))

class Query(BaseModel):
    user_input: str

# Prepare augmented query

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
#index_name = 'hc'
index_name = 'academyzd'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger, a French cryptocurrency company led by CEO Pascal Gauthier. Your primary responsibility is to assist Ledger customer support agents by providing accurate answers to their questions. If a question is unclear or lacks detail, ask for more information instead of making assumptions. If you are unsure of an answer, be honest and seek clarification.

Agents may ask about various Ledger products, including the Ledger Nano S (no battery, low storage), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax (unreleased), Ledger Recover and Ledger Live.
The official Ledger store is located at https://shop.ledger.com/. For authorized resellers, please visit https://www.ledger.com/reseller/ , do not modify or share any other links for these purposes. 

When agents inquire about tokens, crypto or coins supported in Ledger Live , it is crucial to strictly recommend checking the Crypto Asset List link to verify support. 
The link to the Crypto Asset List of supported crypto coins and tokens is: https://support.ledger.com/hc/en-us/articles/10479755500573?docs=true/. Do NOT provide any other links to the list.

VERY IMPORTANT:

- Always mention the source of your information (URL link) when providing answers, such as an official Help Center or Acedemy article or tutorial. If possible, include a direct link to the relevant resource in your response.
- Provide the correct URL link to relevant Help Center or Academy articles or tutorials when responding. Do not share a link if uncertain of its accuracy.
- Direct users who want to learn more about Ledger products or compare devices to https://www.ledger.com/.
- Updating or downloading Ledger Live must always be done via this link: https://www.ledger.com/ledger-live
- Share this list for tips on keeping your recovery phrase safe: https://support.ledger.com/hc/en-us/articles/360005514233-How-to-keep-your-24-word-recovery-phrase-and-PIN-code-safe-?docs=true/


Begin!

"""

# #####################################################


# Define FastAPI app
app = FastAPI()


# Define FastAPI endpoints
@app.get("/")
async def root():
    return {'welcome' : 'You have reached the home route!'}

@app.post('/gpt')
async def react_description(query: Query):
    try:

        res_embed = openai.Embedding.create(
            input=[query.user_input],
            engine=embed_model
        )

        xq = res_embed['data'][0]['embedding']

        res_query = index.query(xq, top_k=5, include_metadata=True)

        contexts = [item['metadata']['text'] for item in res_query['matches']]

        augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"+ "QUESTION: " + "\n\n" +  query.user_input + "? Provide a short answer to the question and make sure to incorporate relevant URL links from the previous CONTEXT. NEVER enclose the links in parentheses. Avoid sharing a link that's not explicitly included in the previous CONTEXT. If you are unable to provide an accurate answer to the question, it is best to honestly acknowledge it and request further information."


        print(augmented_query)

        res = openai.ChatCompletion.create(
            temperature=0.0,
            #model='gpt-4',
            #model="gpt-3.5-turbo-16k",
            model="gpt-3.5-turbo-0613",
            #model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
            ]
        )
        response = res['choices'][0]['message']['content']
        print(response)
        return {'output': response}
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid input")
