#axios.py
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
import uvicorn
from typing import Optional
from time import sleep
# from starlette.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# origins = [
    # "*"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     # allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Define prompt
prompt = """Bot: How can I help you?
User: {{}}
---------------------------------------------
You are a schedule summary. Displays a summary of your schedule in the latest order based on today's date.
"""

# 프롬프트 정의
prompt1 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are a classifier that categorizes the input as either a goal, an event, or a to-do:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
To-Do: Refers to a small task or duty that needs to be accomplished.
When answering, please only answer classification.
"""

# 프롬프트2 정의
prompt2 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are an action type recognizer that categorizes the input as either a create, read, update, or delete:
Create: Includes the act of meeting someone or doing something.
Read: Refers to the act of consuming information or data.
Update: Involves modifying or changing existing information or data.
Delete: Contains the meaning of deleting or making something disappear, Eradication, Elimination.
When answering, please answer the type of action and Say it in a soft tone
"""

# 프롬프트3 정의
prompt3 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You need to know how to distinguish parameters based on the values you enter. First, the default parameter is "title", "startDatime", "endDatime", "location", and "content".
title should contain something symbolic or representative of the value entered by the user. 
startDatetime is An event, schedule, or schedule begins. Must specify "year-month-day-hour-minute" with no words. only datetime.
endDatime is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "year-month-day-hour-minutes" with no words. only datetime.
location means meeting someone or a place. 
I hope the content includes anything other than behavioral and planning. 
image value is always null.
When answering, just answer title, startDatetime, endDatetime, location,content, image in Korean.
"""

prompt4 = """Bot: How can I hlep you?
User: {{$input}}
"""

# 파라미터 추출 함수
def bd_params(input_text):
    print(input_text)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt3}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    # OpenAI API 응답을 바로 FastAPI 엔드포인트의 응답 형식으로 매핑
    response_text = response.choices[0].text.strip()
    # print(response_text)
    # print(f"첫번 째 : {response_text}")
    # print(type(response_text))  # class 'str'
    split_response_text = response_text.split("\n")
    # print(split_response_text)

    response_text = {}
    for _ in split_response_text:
        # print(_)
        bar = _.split(":")
        response_text[f"{bar[0]}"] = bar[1]
    print(response_text)
    return response_text

# 입력을 분류하는 함수
def get_intent(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt1}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()

# 입력을 분류하는 함수 (플랜 추가 관련)
def get_plan_intent(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt2}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()


# 요청과 응답을 위한 Pydantic 모델 정의
class InputText(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    classification: str

# Pydantic 모델 정의
class ParameterExtractionRequest(BaseModel):
    input_text: str

class ParameterExtractionResponse(BaseModel):
    title: Optional[str]
    startDatetime: Optional[str]
    endDatetime: Optional[str]
    location: Optional[str]
    content: Optional[str]

class Goal(BaseModel):
    title: str
    start_date: str
    end_date: str
    location: str
    content: str

# 분류를 위한 엔드포인트 생성
@app.post("/plan_type")
async def plan_type(input_text: InputText):
    input_text = input_text.text
    result = get_intent(input_text)
    return {"classification": result}

# CRUD 분류
@app.post("/plan_crud")
async def plan_crud(input_text: InputText):
    input_text = input_text.text
    result = get_plan_intent(input_text)
    return {"classification": result}

# 파라미터 추출 엔드포인트 생성
@app.post("/extract_parameters")
async def extract_parameters(request_data: ParameterExtractionRequest):  # Request 매개변수 추가
    input_text = request_data.input_text
    extracted_parameters = bd_params(input_text)  # 파라미터 추출 로직을 호출하고 결과를 extracted_parameters에 할당
    return extracted_parameters

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)