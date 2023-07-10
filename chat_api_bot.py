import os
import uuid
import json
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import main
import pinecone
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import re

main.load_dotenv()

env_vars = [
    'OPENAI_API_KEY',
    'ALCHEMY_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT',
]

os.environ.update({key: os.getenv(key) for key in env_vars})
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
openai.api_key=os.environ['OPENAI_API_KEY']


class Query(BaseModel):
    user_input: str

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

# Prepare augmented query

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
index_name = 'hc'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger, a French cryptocurrency company led by CEO Pascal Gauthier. Your primary responsibility is to assist Ledger users by providing accurate answers to their questions. If a question is unclear or lacks detail, ask for more information instead of making assumptions. If you are unsure of an answer, be honest and seek clarification.

Users may ask about various Ledger products, including the Ledger Nano S (no battery, low storage), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax (unreleased), Ledger Recover and Ledger Live.
The official Ledger store is located at https://shop.ledger.com/. The Ledger Recover White Paper is located at https://github.com/LedgerHQ/recover-whitepaper . For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When users inquire about tokens, crypto or coins supported in Ledger Live , it is crucial to strictly recommend checking the Crypto Asset List link to verify support: https://support.ledger.com/hc/en-us/articles/10479755500573?docs=true/. Do NOT provide any other links to the list.

VERY IMPORTANT:

- If the query is not about Ledger products, disregard the CONTEXT. Respond courteously and invite any Ledger-related questions.
- When responding to a question, include a maximum of two URL links from the provided CONTEXT, choose the most relevant.
- Do not share URLs if none are mentioned within the CONTEXT.
- Always present URLs as plain text, never use markdown formatting.
- If a user ask to speak to a human agent, invite them to contact us via this link: https://support.ledger.com/hc/en-us/articles/4423020306705-Contact-Us?support=true
- If a user reports being victim of a scam or unauthorized crypto transactions, empathetically acknowledge their situation, promptly connect them with a live agent, and share this link for additional help: https://support.ledger.com/hc/en-us/articles/7624842382621-Loss-of-funds?support=true.
- Direct users who want to learn more about Ledger products or compare devices to https://www.ledger.com/.
- Updating or downloading Ledger Live must always be done via this link: https://www.ledger.com/ledger-live
- Share this list for tips on keeping your recovery phrase safe: https://support.ledger.com/hc/en-us/articles/360005514233-How-to-keep-your-24-word-recovery-phrase-and-PIN-code-safe-?docs=true/


Begin!

"""

# #####################################################


# Define FastAPI app
app = FastAPI()

last_response = None


# Define FastAPI endpoints
@app.get("/")
async def root():
    return {'welcome' : 'You have reached the home route!'}

@app.get("/")
async def root():
    return {'welcome' : 'You have reached the home route!'}


@app.post('/gpt')
async def react_description(query: Query):
    # Define get_gpt_response() inside your app logic
    async def get_gpt_response(query):
        global last_response
        res_embed = openai.Embedding.create(
            input=[query.user_input],
            engine=embed_model
        )

        xq = res_embed['data'][0]['embedding']

        res_query = index.query(xq, top_k=5, include_metadata=True)
        print(res_query)

        contexts = [item['metadata']['text'] for item in res_query['matches'] if item['score'] > 0.78]
        
        prev_response_line = f"YOUR PREVIOUS RESPONSE: {last_response}\n\n-----\n\n" if last_response else ""
        
        augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + prev_response_line + "USER QUESTION: " + "\n\n" + '"' + query.user_input + '" ' + "\n\n" + "YOUR RESPONSE: "

        
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
        

        print(response)
        print(type(response))
        
        docs = [Document(page_content=response)]
        return docs

    try:
        global last_response
        #response = await get_gpt_response(query)
        docs = await get_gpt_response(query)


################################## REACT AGENT #################################

        # tools = [
        #     Tool(
        #         name = "Knowledge Base",
        #         func=lambda *args, **kwargs: response,
        #         description="Always use to answer technical questions about Ledger products."
        #     ),
        # ]

        # memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     k=5, 
        #     return_messages=True)
        # print(memory)

        # llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)

        # agent_chain = initialize_agent(

        #     tools, 
        #     llm, 
        #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        #     verbose=True, 
        #     memory=memory,
        #     max_iterations=3,
                                    
        # )

        # chat = agent_chain.run(input=query.user_input)

##################### QA CHAIN #######################

        last_response_text = last_response if last_response else " "

        template = """You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger, a French cryptocurrency company led by CEO Pascal Gauthier. 
        Your primary responsibility is to assist Ledger users by providing accurate answers to their questions.

        Your intuition is a powerful tool in resolving issues. Pay attention to what it suggests, but remember it's not infallible. 
        If a solution doesn't immediately present itself, don't hesitate to ask clarifying questions. They can often lead you towards the answer.

        YOUR INTUITION:
        {context}

        ______________

        YOUR PREVIOUS ANSWER: """ + last_response_text + """
        USER QUESTION: {human_input}
        YOUR RESPONSE:"""


        prompt = PromptTemplate(
            input_variables=["human_input", "context"], template=template
        )

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", k=2)
        print(memory)

        chain = load_qa_chain(
            OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0), chain_type="stuff", memory=memory, prompt=prompt, verbose=True
        )

        query = query.user_input

        chat  = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)\
        
        last_response = chat['output_text']

        return {'output': chat['output_text']}
        #return {'output': chat}
    
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid input")
    
############### START COMMAND ##########

#   uvicorn memory_api_bot:app --reload --port 8008
#   sudo uvicorn api_bot:app --port 80 --host 0.0.0.0


########VM Service Commands#####

# sudo nano /etc/nginx/sites-available/myproject
# sudo systemctl restart nginx
# sudo systemctl stop nginx

# sudo nano /etc/systemd/system/api_bot.service
# sudo systemctl daemon-reload
# sudo systemctl start api_bot to start the service.
# sudo systemctl stop api_bot to stop the service.
# sudo systemctl restart api_bot to restart the service (after modifying the code for example)
# sudo systemctl status api_bot to check the status of the service.
# journalctl -u api_bot.service -e to check logs
