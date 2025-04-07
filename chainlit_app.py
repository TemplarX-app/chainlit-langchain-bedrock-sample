import chainlit as cl
from typing import Dict, Optional
from chainlit.input_widget import Select, Switch, Slider
import time
import asyncio

from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws.agents import BedrockAgentsRunnable
from langchain.chains import RetrievalQA

import boto3
from botocore.config import Config
import botocore.exceptions
import sys
import os


boto3_config = Config(
    region_name='us-east-1',  # Replace with your region
    retries={
        'max_attempts': 3,     # Maximum number of retries
        'mode': 'standard',    # Use standard retry mode instead of legacy
    },
    connect_timeout=10,    # Connection timeout in seconds
    read_timeout=10,      # Read timeout in seconds
    max_pool_connections=10,  # Maximum connection pool size
)

module_path = ".."
sys.path.append(os.path.abspath(module_path))
region = 'us-east-1'
bedrock_client = boto3.client('bedrock-runtime',
                              region_name=region,
                              config=boto3_config)
bedrock_agent_client = boto3.client('bedrock-agent-runtime', 
                                    region_name=region, 
                                    config=boto3_config)

# USE AURORA POSTGRES
KNOWLEDGE_BASE_ID = 'your knowledge base id'
# USE OPENSEARCH
# KNOWLEDGE_BASE_ID = 'your knowledge base id'
GUARDRAIL_ID =  'your guardrail id'
GUARDRAIL_VERSION = 'DRAFT'

# Retry mechanism for Aurora DB auto-pause resumption
async def retry_on_aurora_resuming(operation_func, max_retries=10, initial_backoff=5, backoff_multiplier=1.5):
    """
    Retry an async function when Aurora DB is resuming from auto-pause or is stopped.
    
    Args:
        operation_func: The async function to retry
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial wait time in seconds
        backoff_multiplier: Factor by which to increase backoff time after each retry
        
    Returns:
        The result of the operation function
    """
    retries = 0
    backoff = initial_backoff
    
    while True:
        try:
            return await operation_func()
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = str(e)
            
            # Check if error is due to Aurora resuming or being stopped
            if (error_code == 'ValidationException' and 
                ('resuming after being auto-paused' in error_message or 
                 'is in stopped state' in error_message) and 
                retries < max_retries):
                
                print(f"Aurora DB is not ready. Retrying in {backoff} seconds... (Attempt {retries + 1}/{max_retries})")
                await asyncio.sleep(backoff)
                retries += 1
                backoff *= backoff_multiplier
            else:
                # If it's a different error or we've exceeded retries, re-raise
                raise

# Create the standard retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=KNOWLEDGE_BASE_ID,
    region_name=region,
    client=bedrock_agent_client,
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 3
        }
    },
    generation_config={
        "guardrailConfiguration": {
            "guardrailIdentifier": GUARDRAIL_ID,
            "guardrailVersion": GUARDRAIL_VERSION
        }
    },
    config=boto3_config
)


# @cl.oauth_callback
# def oauth_callback(
#   provider_id: str,
#   token: str,
#   raw_user_data: Dict[str, str],
#   default_user: cl.User,
# ) -> Optional[cl.User]:
#   """
#   This function is called after the user has been authenticated.
#   TODO: Comment it out to test in LOCAL
#   """
#   return default_user

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "@dm!n"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Medical Insurance Bot",
            markdown_description="This is a medical & insurance bot",
            icon="/public/aws.svg",
            starters=[
                cl.Starter(
                    label="Why do I need health insurance?",
                    message="Can you help me to understand why do I need health insurance?.",
                    icon="/public/insurance-user-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="What is the symptom of common cold?",
                    message="Could you please tell me more about the symptom of common cold?",
                    icon="/public/first-aid-kit-doctor-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="Tips for losing weight",
                    message="Tips for losing weight",
                    icon="https://picsum.photos/230",
                ),
            ],
        )
    ]


@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    chat_profile = cl.user_session.get("chat_profile")
    settings = None  # Initialize settings with a default value

    if chat_profile=='Medical Insurance Bot':
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="Model",
                    values=["Claude-3.7-Sonnet", "Amazon-Nova-Pro"],
                    initial_index=0,
                ),
                Switch(
                    id="UseKnowledgeBase",
                    label="Enable Knowledge Base",
                    initial=False,
                ),
                Switch(
                    id="UseAgent",
                    label="Enable Agents",
                    initial=False,
                ),
                Slider(
                    id="Temperature",
                    label="Temperature",
                    initial=0.7,
                    min=0,
                    max=1,
                    step=0.1,
                ),
                Slider(
                    id="MaxTokens",
                    label="Max Tokens",
                    initial=1000,
                    min=100,
                    max=4000,
                    step=100,
                ),
                Slider(
                    id="topP",
                    label="Top P",
                    initial=0.9,
                    min=0,
                    max=1,
                    step=0.1,
                ),
            ]
        ).send()
    if settings:  # Only call setup_agent if settings is available
        await setup_agent(settings)
        

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

    prompt_insurance = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a health insurance agent and medical expert that is willing to help the user answer questions related to health insurance and common medical topic topics.
          Your focus is not just selling the insurance product but also solving user's concerns related to healthcare and symptom of disease. Do not do hard selling, be humble.
          Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
          Use your internal knowledge first, if you do not have the answer, use knowledge bases, if you still do not know the answer, do not hallucinate. 
         
         Guidelines:
          - Only provide factual medical information from reliable sources
          - Do not give specific medical advice or diagnoses
          - Refer users to healthcare professionals for medical concerns
          - Do not discuss sensitive personal health information
          - Focus on general insurance and healthcare topics
          - Avoid answering questions unrelated to Insurance and Medical
          - Never show table name in the response if users asks about insurance
         
         {context}
         """),
        ("human", "{question}"),
    ])

    """
    ChatPromptTemplate Explained:
    system: This is used to baseline the AI assistant. 
            If you want to set some behaviour of your assistant. 
            This is the place to do it.
    human: All the user inputs are to stored in this
    """

    guardrails={
                'guardrailIdentifier': GUARDRAIL_ID,
                'guardrailVersion': GUARDRAIL_VERSION,
                'trace': "enabled"}
    
    if settings["Model"] == "Claude-3.7-Sonnet":
        chat_model=ChatBedrockConverse(
            model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region_name=region,
            temperature=float(settings["Temperature"]),
            max_tokens=int(settings["MaxTokens"]),
            top_p=float(settings["topP"]),
            guardrail_config=guardrails
        )

    elif settings["Model"] == "Amazon-Nova-Pro":
        chat_model=ChatBedrockConverse(
            model="amazon.nova-pro-v1:0",
            region_name=region,
            temperature=float(settings["Temperature"]),
            max_tokens=int(settings["MaxTokens"]),
            top_p=float(settings["topP"]),
            guardrail_config=guardrails
        )

    if settings["UseKnowledgeBase"]:
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_insurance
            }
        )
        runnable = qa_chain

    elif settings["UseAgent"]:
        bedrock_agent_runnable = BedrockAgentsRunnable(
            agent_id="your agent id",
            agent_alias_id="your agent alias id",
            client=bedrock_agent_client,
            guardrail_configuration=guardrails
        )
        runnable = bedrock_agent_runnable

    else:
        runnable = prompt_insurance | chat_model | StrOutputParser()

    cl.user_session.set("runnable", runnable)
    

@cl.on_message
async def main(message: cl.Message):
    start = time.time()
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")
    loading_msg = None

    if isinstance(runnable, RetrievalQA):
        # Show a loading message while we wait for the database
        loading_msg = cl.Message(content="Retrieving information...")
        await loading_msg.send()
        
        # Use retry mechanism for RetrievalQA
        async def retrieval_operation():
            # This is where we need to handle the Aurora DB resumption error
            try:
                return await runnable.ainvoke(
                    {"query": message.content},
                    config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
                )
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = str(e)
                
                if (error_code == 'ValidationException' and 
                    ('resuming after being auto-paused' in error_message or 
                     'is in stopped state' in error_message)):
                    # Re-raise to be caught by the retry mechanism
                    raise
                else:
                    # For other errors, wrap them to provide more context
                    raise Exception(f"Error during retrieval: {str(e)}")
        
        try:
            # Apply retry mechanism to the retrieval operation
            response = await retry_on_aurora_resuming(retrieval_operation)
            
            # Remove the loading message
            if loading_msg:
                await loading_msg.remove()
            
            answer = response['result']
            source_documents = response['source_documents']

            # Send the main answer
            msg.content = answer
            await msg.send()

            # Create side elements for sources
            if 'Sorry, the model cannot answer this question.' in answer:
                pass
            elif source_documents:
                elements = []
                for i, doc in enumerate(source_documents, 1):
                    source_content = f"Content: {doc.page_content}\n"
                    # Create an expandable text element for each source
                    elements.append(
                        cl.Text(
                            name=f"Source",
                            content=source_content,
                            display="side"
                        )
                    )
                
                # Send a new message with the source elements
                await cl.Message(
                    content="📚 Reference Source (click to show):",
                    elements=elements
                ).send()
        except Exception as e:
            # Remove the loading message
            if loading_msg:
                await loading_msg.remove()
                
            # For debugging purposes, log the error but don't show it to the user
            print(f"Error occurred: {str(e)}")
            
            # Show a friendly message instead of the error
            await cl.Message(content="I'm having trouble connecting to the knowledge base. Please try again in a moment.").send()

    elif isinstance(runnable, BedrockAgentsRunnable):
        # Show a loading message while we wait for the database
        loading_msg = cl.Message(content="Processing your request...")
        await loading_msg.send()
        
        # Use retry mechanism for BedrockAgentsRunnable
        async def agent_operation():
            try:
                return await runnable.ainvoke(
                    {
                        "input": message.content
                    },
                    config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
                )
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = str(e)
                
                if (error_code == 'ValidationException' and 
                    ('resuming after being auto-paused' in error_message or 
                     'is in stopped state' in error_message)):
                    # Re-raise to be caught by the retry mechanism
                    raise
                else:
                    # For other errors, wrap them to provide more context
                    raise Exception(f"Error during agent operation: {str(e)}")
        
        try:
            # Apply retry mechanism to the agent operation
            response = await retry_on_aurora_resuming(agent_operation)
            
            # Remove the loading message
            if loading_msg:
                await loading_msg.remove()
                
            answer = response.return_values['output']
            msg.content = answer
            await msg.send()
        except Exception as e:
            # Remove the loading message
            if loading_msg:
                await loading_msg.remove()
                
            # For debugging purposes, log the error but don't show it to the user
            print(f"Error occurred: {str(e)}")
            
            # Show a friendly message instead of the error
            await cl.Message(content="I'm having trouble processing your request. Please try again in a moment.").send()
    
    else:
        # Case when knowledge base is disabled
        msg = cl.Message(content="")

        try:
            async for chunk in runnable.astream(
                {"question": message.content, "context": ""},  # Provide empty context when KB is disabled
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)
        except Exception as e:
            # For debugging purposes, log the error but don't show it to the user
            print(f"Error occurred: {str(e)}")
            
            # Show a friendly message instead of the error
            await cl.Message(content="I'm having trouble generating a response. Please try again in a moment.").send()

    # Only show response time if we successfully completed the operation
    if msg.content:
        await cl.Message(content=f'Response time: {time.time() - start:.2f}s').send()
