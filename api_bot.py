import os
import uuid
import json
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from web3 import Web3
import pinecone
import openai
from fastapi import FastAPI
from pydantic import BaseModel

import re

load_dotenv()

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
index_name = 'hc'
#index_name = 'academyzd'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger, a French cryptocurrency company led by CEO Pascal Gauthier. Your primary responsibility is to assist Ledger users by providing accurate answers to their questions. If a question is unclear or lacks detail, ask for more information instead of making assumptions. If you are unsure of an answer, be honest and seek clarification.

Agents may ask about various Ledger products, including the Ledger Nano S (no battery, low storage), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax (unreleased), Ledger Recover and Ledger Live.
The official Ledger store is located at https://shop.ledger.com/. The Ledger Recover White Paper is located at https://github.com/LedgerHQ/recover-whitepaper . For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When agents inquire about tokens, crypto or coins supported in Ledger Live , it is crucial to strictly recommend checking the Crypto Asset List link to verify support. 
The link to the Crypto Asset List of supported crypto coins and tokens is: https://support.ledger.com/hc/en-us/articles/10479755500573?docs=true/. Do NOT provide any other links to the list.

VERY IMPORTANT:

- If the query is not about Ledger products, disregard the CONTEXT. Respond courteously and invite any Ledger-related questions.
- When answering a question, make sure to incorporate only the most relevant URL link from the CONTEXT in your response. 
- Only share URLs that are explicitly mentioned within the previous CONTEXT.
- If the provided CONTEXT does not include URLs, do not include any in your response.
- Always present URLs as plain text, never use markdown formatting.
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

        contexts = [item['metadata']['text'] for item in res_query['matches'] if item['score'] > 0.8]

        augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"+ "QUESTION: " + "\n\n" + '"' + query.user_input + '" ' + "Please provide an answer to the question."
        


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

############### START COMMAND ##########

#   uvicorn api_bot:app --reload --port 8008
