import os
import boto3
import json
from dotenv import main
import pinecone
from helicone.openai_async import openai
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, TypeAdapter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import BackgroundTasks
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from nostril import nonsense
import tiktoken
import re
import time


# Secret Management
def access_secret_parameter(parameter_name):
    ssm = boto3.client('ssm', region_name='eu-west-3')
    response = ssm.get_parameter(
        Name=parameter_name,
        WithDecryption=True
    )
    return response['Parameter']['Value']

env_vars = {
    'ACCESS_KEY_ID': access_secret_parameter('ACCESS_KEY_ID'),
    'SECRET_ACCESS_KEY': access_secret_parameter('SECRET_ACCESS_KEY'),
    'BACKEND_API_KEY': access_secret_parameter('BACKEND_API_KEY'),
    'OPENAI_API_KEY': access_secret_parameter('OPENAI_API_KEY'),
    'HELICONE_API_KEY': access_secret_parameter('HELICONE_API_KEY'),
    'PINECONE_API_KEY': access_secret_parameter('PINECONE_API_KEY'),
    'PINECONE_ENVIRONMENT': access_secret_parameter('PINECONE_ENVIRONMENT')
}

# Set up boto3 session with AWS credentials
boto3.setup_default_session(
    aws_access_key_id=os.getenv('ACCESS_KEY_ID', env_vars['ACCESS_KEY_ID']),
    aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY', env_vars['SECRET_ACCESS_KEY']),
    region_name='eu-west-3'
)

os.environ.update(env_vars)

#########Initialize backend API keys ######

server_api_key=os.environ['BACKEND_API_KEY']
API_KEY_NAME="Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header or api_key_header.split(' ')[1] != server_api_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header

class Query(BaseModel):
    user_input: str
    user_id: str

# Initialize Pinecone
openai.api_key=os.environ['OPENAI_API_KEY']
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
index_name = 'personal'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

with open('system_prompt.txt', 'r') as sys_file:
    primer = sys_file.read()

# #####################################################

# Email address  detector
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
def find_emails(text):
    return re.findall(email_pattern, text)

# Address filter:
ETHEREUM_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b'
BITCOIN_ADDRESS_PATTERN = r'\b(1|3)[1-9A-HJ-NP-Za-km-z]{25,34}\b|bc1[a-zA-Z0-9]{25,90}\b'
LITECOIN_ADDRESS_PATTERN = r'\b(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}\b'
DOGECOIN_ADDRESS_PATTERN = r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b'
XRP_ADDRESS_PATTERN = r'\br[a-zA-Z0-9]{24,34}\b'
COSMOS_ADDRESS_PATTERN = r'\bcosmos[0-9a-z]{38,45}\b'
SOLANA_ADDRESS_PATTERN= r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'


tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_user_id(request: Request):
    try:
        body = TypeAdapter(Query).validate_python(request.json())
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

# Initialize user state and periodic cleanup function
user_states = {}
TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours

def periodic_cleanup(background_tasks: BackgroundTasks):
    while True:
        cleanup_expired_states()
        time.sleep(TIMEOUT_SECONDS)

# Invoke periodic cleanup
@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(periodic_cleanup)

# Handle cleanup crashes gracefully
def cleanup_expired_states():
    try:
        current_time = time.time()
        expired_users = [
            user_id for user_id, state in user_states.items()
            if current_time - state['timestamp'] > TIMEOUT_SECONDS
        ]
        for user_id in expired_users:
            del user_states[user_id]
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Define FastAPI endpoints
@app.get("/")
async def root():
    return {"welcome": "You've reached the home route!"}

@app.get("/_health")
async def health_check():
    return {"status": "OK"}

@app.get("/_index")
async def pinecone_index():
    return {"index": index_name}

@app.post('/gpt')
@limiter.limit("50/minute")
async def react_description(query: Query, request: Request, api_key: str = Depends(get_api_key)):
    user_id = query.user_id
    user_input = query.user_input.strip()

    if user_id not in user_states:
        user_states[user_id] = {
            'previous_queries': [],
            'timestamp': time.time()
        }

    if not user_input or nonsense(user_input):
        print('Nonsense detected!')
        return {'output': "I'm sorry, I cannot understand your question, and I can't assist with questions that include cryptocurrency addresses. Could you please provide more details or rephrase it without the address? Remember, I'm here to help with any Ledger-related inquiries."}

    if re.search(ETHEREUM_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(BITCOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(LITECOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(DOGECOIN_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(COSMOS_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(SOLANA_ADDRESS_PATTERN, user_input, re.IGNORECASE) or \
           re.search(XRP_ADDRESS_PATTERN, user_input, re.IGNORECASE):
        return {'output': "I'm sorry, but I can't assist with questions that include cryptocurrency addresses. Please remove the address and ask again."}

    if re.search(email_pattern, user_input):
        return {
            'output': "I'm sorry, but I can't assist with questions that include email addresses. Please remove the address and ask again."
        }

    else:

        try:
            # Define Retrieval
            async def retrieve(query, contexts=None):
                res_embed = openai.Embedding.create(
                    input=[user_input],
                    engine=embed_model
                )
                xq = res_embed['data'][0]['embedding']
                res_query = index.query(xq, top_k=2, include_metadata=True)
                # Filter items with score > 0.77 and sort them by score
                sorted_items = sorted([item for item in res_query['matches'] if item['score'] > 0.77], key=lambda x: x['score'])

                # Construct the contexts
                contexts = []
                for idx, item in enumerate(sorted_items):
                    context = item['metadata']['text']
                    context += "\nLearn more: " + item['metadata'].get('source', 'N/A')
                    contexts.append(context)
                previous_conversation = '\n'.join([f"User: {query}\nAssistant: {response}"for query, response in user_states[user_id].get('previous_queries', [])])
                augmented_query = "CONTEXT: " + "\n\n" + "\n\n".join(contexts) + "\n\n-----\n\n" + "CHAT HISTORY: \n" +  previous_conversation + "\n\n-----\n\n" + "User: " + user_input + "\n" + "Assistant: "

                return augmented_query

            # Start Retrieval
            augmented_query = await retrieve(user_input)
            print(augmented_query)

            # Request and return OpenAI RAG
            async def rag(query, contexts=None):
                print("RAG > Called!")
                res = openai.ChatCompletion.create(
                    temperature=0.0,
                    model='gpt-4',
                    #model="gpt-3.5-turbo-0613",
                    messages=[
                        {"role": "system", "content": primer},
                        {"role": "user", "content": augmented_query}
                    ]
                )
                reply = res['choices'][0]['message']['content']
                return reply

            # Start RAG
            response = await rag(augmented_query)

            # Count tokens
            async def print_total_price(response, augmented_query, primer):
                count_response = tiktoken_len(response)
                count_query = tiktoken_len(augmented_query)
                count_sysprompt = tiktoken_len(primer)
                total_input_tokens = count_sysprompt + count_query
                total_output_tokens = count_response
                final_count = total_output_tokens + total_input_tokens
                total_price = str(((total_input_tokens / 1000) * 0.03) + ((total_output_tokens / 1000) * 0.06))
                return(total_price)

            total_cost = await print_total_price(response=response, augmented_query=augmented_query, primer=primer)
            print("Total price for GPT4 call: " + total_cost + " $USD")


            # Save the response to a thread
            user_states[user_id] = {
                'previous_queries': user_states[user_id].get('previous_queries', []) + [(user_input, response)],
                'timestamp': time.time()
            }
            print (user_states)
            print(response)
            return {'output': response}

        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail="Invalid input")

#####RUN COMMADND########
#  uvicorn slack_bot:app --port 8000
# in Google Cloud
# sudo uvicorn slack_bot:app --port 80 --host 0.0.0.0

########VM Service Commands#####

# sudo nano /etc/nginx/sites-available/myproject
# sudo systemctl restart nginx
#sudo systemctl stop nginx

# sudo nano /etc/systemd/system/slack_bot.service
# sudo systemctl daemon-reload
# sudo systemctl start slack_bot to start the service.
# sudo systemctl stop slack_bot to stop the service.
# sudo systemctl restart slack_bot to restart the service (after modifying the code for example)
# sudo systemctl status slack_bot to check the status of the service.
# journalctl -u slack_bot.service -e to check logs