import os
import uuid
import json
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import main
import pinecone
import openai
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, parse_obj_as
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from nostril import nonsense

import re

from google.cloud import secretmanager

def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')


env_vars = {
    'OPENAI_API_KEY': access_secret_version('slack-bot-391618', 'OPENAI_API_KEY', 'latest'),
    'PINECONE_API_KEY': access_secret_version('slack-bot-391618', 'PINECONE_API_KEY', 'latest'),
    'PINECONE_ENVIRONMENT': access_secret_version('slack-bot-391618', 'PINECONE_ENVIRONMENT', 'latest'),
    'BACKEND_API_KEY': access_secret_version('slack-bot-391618', 'BACKEND_API_KEY', 'latest')

}


os.environ.update(env_vars)

openai.api_key=os.environ['OPENAI_API_KEY']
server_api_key=os.environ['BACKEND_API_KEY'] 

#### INITIALIZE API ACCESS KEY #####

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header.split(' ')[1] != server_api_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header


class Query(BaseModel):
    user_input: str
    user_id: str 


# Prepare augmented query

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
#index_name = 'hc'
index_name = 'academyzd'

index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger. Your primary responsibility is to assist Ledger users by providing accurate answers to their questions.

Users may ask about various Ledger products, including the Ledger Nano S (no battery, low storage), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax, and Ledger Live.
The official Ledger store is located at https://shop.ledger.com/. For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When users inquire about tokens, crypto or coins supported in Ledger Live, it is crucial to strictly recommend checking the Crypto Asset List link to verify support: https://support.ledger.com/hc/en-us/articles/10479755500573?docs=true/. Do NOT provide any other links to the list.

VERY IMPORTANT:

- Use the CONTEXT and CHAT HISTORY to answer users questions
- When responding to a question, include a maximum of two URL links from the provided CONTEXT, choose the most relevant.
- If the question is unclear or not about Ledger products, disregard the CONTEXT and invite any Ledger-related questions using this exact response: "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."
- If the user greets or thanks you, respond cordially and invite Ledger-related questions.
- Always present URLs as plain text, never use markdown formatting.
- If a user asks to speak to a human agent, invite them to contact us via this link: https://support.ledger.com/hc/en-us/articles/4423020306705-Contact-Us?support=true
- If a user reports being victim of a scam or unauthorized crypto transactions, empathetically acknowledge their situation, promptly connect them with a live agent, and share this link for additional help: https://support.ledger.com/hc/en-us/articles/7624842382621-Loss-of-funds?support=true.
- If a user needs to reset their device, they must always ensure they have their recovery phrase on hand before proceeding with the reset.
- Updating or downloading Ledger Live must always be done via this link: https://www.ledger.com/ledger-live
- If asked about Ledger Stax, inform the user it's not yet released, but pre-orderers will be notified via email when ready to ship. Share this link for more details: https://support.ledger.com/hc/en-us/articles/7914685928221-Ledger-Stax-FAQs.
- The Ledger Recover service is not available just yet. When it does launch, keep in mind that it will be entirely optional. Even if you update your device firmware, it will NOT automatically activate the Recover service. Learn more: https://support.ledger.com/hc/en-us/articles/9579368109597-Ledger-Recover-FAQs
- If you see the error "Something went wrong - Please check that your hardware wallet is set up with the recovery phrase or passphrase associated to the selected account", it's likely your Ledger's recovery phrase doesn't match the account you're trying to access.

Begin!

"""

# #####################################################

def get_user_id(request: Request):
    try:
        body = parse_obj_as(Query, request.json())
        user_id = body.user_id
        return user_id
    except Exception as e:
        return get_remote_address(request)

# Define FastAPI app
app = FastAPI()

# Define limiter
limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests, please try again in a minute."},
    )

user_states = {} #New
print(user_states)

# Define FastAPI endpoints
@app.get("/")
async def root():
    return {'welcome' : 'You have reached the home route!'}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post('/gpt')
@limiter.limit("20/minute")
def react_description(query: Query, request: Request, api_key: str = Depends(get_api_key)):
    print(f"Received request with data: {query.dict()}")
    user_id = query.user_id  # New - Get the user ID from the request
    user_input = query.user_input.strip()
    if user_id not in user_states:  # New -  Initialize a new state if necessary
        user_states[user_id] = None #New
    last_response = user_states[user_id]
    if not query or nonsense(query):
        print('Nonsense detected!')
        return {'output': "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."}
    else:
        try:
            res_embed = openai.Embedding.create(
                input=[user_input],
                engine=embed_model
            )
    
            xq = res_embed['data'][0]['embedding']
    
            res_query = index.query(xq, top_k=3, include_metadata=True)
            print(res_query)
    
            contexts = [(item['metadata']['text'] + "\nSource: " + item['metadata'].get('source', 'N/A')) for item in res_query['matches'] if item['score'] > 0.75]
    
            # If there's a previous response, include it in the augmented query
            prev_response_line = f"Assistant: {last_response}\n" if last_response else ""
    
            augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"+ "CHAT HISTORY: \n" + prev_response_line + "User:" + user_input + "\n" + "Assistant: "
    
            print(augmented_query)
    
            res = openai.ChatCompletion.create(
                temperature=0.0,
                model='gpt-4',
                messages=[
                    {"role": "system", "content": primer},
                    {"role": "user", "content": augmented_query}
                ]
            )
            response = res['choices'][0]['message']['content']
    
            # Save the response to the global variable
            #last_response = response
            user_states[user_id] = response #New
    
            print(response)
            return {'output': response}
        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail="Invalid input")
