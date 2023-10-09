#axios.py
from fastapi import FastAPI, HTTPException, Query
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
from fastapi import Form
from fastapi.responses import JSONResponse
# from starlette.middleware.cors import CORSMiddleware

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
# FastAPI 애플리케이션 인스턴스를 생성합니다.
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

# .env 파일에서 OpenAI API 키를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# 다양한 사용 사례에 대한 프롬프트를 정의합니다.
prompt1 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are a classifier that categorizes the input as either a goal, an event, or a to-do:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
To-Do: Refers to a small task or duty that needs to be accomplished.
When answering, speak Naturally and when you reply, you must specify either a goal, event or todo.
"""

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

prompt3 = """
User: {{$input}}
---------------------------------------------
You need to know how to distinguish parameters based on the values you enter. First, the default parameter is "title", "startDatime", "endDatime", "location", "content", "image" and "isCompleted".
title should contain something symbolic or representative of the value entered by the user. 
startDatetime is An event, schedule, or schedule begins. Must specify "year-month-day-hour-minute" with no words. only datetime.
endDatime is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "year-month-day-hour-minutes" with no words. only datetime.
location means meeting someone or a place. 
I hope the content includes anything other than behavioral and planning. 
image value is always null.
isCompleted value begins with false unless users specify when they say or mention it is done or complete. They they say or mention it is complete, change the value from false to true.
When answering, ignore special characters, symbols and blank spaces. Just answer title, startDatetime, endDatetime, location,content, image, and isCompleted.
"""

prompt4 = """
User: {{$input}}
Recognize the date of the user's input and bring only the date value to yyyy-mm-dd. The exact date and time
https://vclock.kr/time/ Please check the url,find the exact date and just answer the date without words.
And when you recognize the date, year is always 2023.
"""

# 첫 번째 프롬프트에 대한 OpenAI를 사용하여 입력을 분류하는 함수
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

# 두 번째 프롬프트에 대한 OpenAI를 사용하여 입력을 분류하는 함수
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

# 세 번째 프롬프트에 대한 OpenAI를 사용하여 파라미터를 추출하는 함수
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
    # OpenAI의 응답에서 텍스트를 추출하고 앞뒤 공백을 제거합니다.
    response_text = response.choices[0].text.strip()
    # 추출된 텍스트를 줄 단위로 나눕니다.
    split_response_text = response_text.split("\n")
    response_text = {}
    print(response_text)
    # 각 줄을 처리하여 파라미터를 추출합니다.
    for _ in split_response_text:
    # 줄을 콜론(:)을 기준으로 나눕니다.
        bar = _.split(":")
    # 적어도 두 개의 부분으로 나눠진 경우에만 처리합니다.
        if len(bar) >= 2:
    # 첫 번째 부분을 키로, 두 번째 부분을 값으로 사용하여 딕셔너리에 추가합니다.
            response_text[f"{bar[0]}"] = bar[1]
    print(response_text)
    return response_text

# 네 번째 프롬프트에 대한 OpenAI를 사용하여 날짜를 인식하는 함수
def get_date(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt4}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    date_str = response.choices[0].text.strip()

    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        return date_str

def get_token(token):
    response = requests.post(
        url='http://43.201.211.135:3000/account/login',
        data={'token': token}
    )
    
    # 외부 서버에서의 응답을 확인하고 적절한 처리를 수행
    if response.status_code == 200:
        return response.json()  # 외부 서버에서 반환한 JSON을 그대로 클라이언트에게 반환
    else:
        # 실패했을 경우 클라이언트에게 에러 메시지를 반환
        return JSONResponse(content={"message": "외부 서버에 요청 중 오류가 발생했습니다."}, status_code=500)


# 요청 및 응답을 위한 Pydantic 모델을 정의합니다.
class InputText(BaseModel):
    text: str

# Pydantic 모델 정의
class ParameterExtractionRequest(BaseModel):
    input_text: str

# FastAPI 데코레이터를 사용하여 엔드포인트를 생성합니다.
@app.post("/receive_token")
async def receive_token(token: str = Form(...)):
    print('받은 토큰:', token)
    return {"message": "토큰을 성공적으로 받았습니다."}

# FastAPI 데코레이터를 사용하여 엔드포인트를 생성합니다.
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
@app.post("/create/extract_parameters")
async def extract_parameters(request_data: ParameterExtractionRequest):  # Request 매개변수 추가
    input_text = request_data.input_text
    extracted_parameters = bd_params(input_text)  # 파라미터 추출 로직을 호출하고 결과를 extracted_parameters에 할당
    return extracted_parameters

# FastAPI 서버에서 /get_date 엔드포인트에서
@app.post("/get_date")
async def get_date(input_data: InputText, token: str = Form(...)):
    result = get_date(input_data.text)
    return {"date": result, "token": token}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# @app.post("/get_date")
# async def get_date_handler(token: str = Form(...)):
#     # get_token function is called to get the Python token
#     python_token = get_token(token)

#     # The obtained Python token is used to call the get_date function
#     result = get_date(python_token)

#     return {"result": result}

# UVicorn을 사용하여 FastAPI 애플리케이션을 실행합니다.



# @app.post("/get_date")
# async def get_date_handler(token: str = Form(...)):
#     result = get_date(token)
#     return {"result": result}   

# @app.post("/get_date")
# async def get_date_handler(token: str = Form(...)):
#     # get_token 함수를 호출하여 토큰을 가져옴
#     python_token = get_token(token)

#     # 가져온 토큰을 사용하여 get_date 함수 호출
#     result = get_date(python_token)
#     return {"result": result} 
