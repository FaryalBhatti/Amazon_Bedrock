#from langchain.llms import Bedrock
#from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate 
import boto3 
import streamlit as st


#bedrock client

bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

llm = ChatBedrock(
    model_id= model_id,
    client= bedrock_client,
    model_kwargs={"temperature": 0.9}
)



def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot. You are in {language}.\n\n{user_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response=bedrock_chain({'language':language, 'user_text':user_text})

    return response



st.title("Amazon Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "spanish", "hindi"])

if language:
    user_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)


if user_text:
    response = my_chatbot(language,user_text)
    st.write(response['text'])