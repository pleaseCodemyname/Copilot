#axios.py
from fastapi import FastAPI,HTTPException, Query
from pydantic import BaseModel
from typing import List
import requests
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import uvicorn
import json


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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

#프롬프트3 정의
prompt3 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You need to know how to distinguish parameters based on the values you enter. First, the default parameter is "title", "startDatime", "endDatime", "location", and "content".
Title should contain something symbolic or representative of the value entered by the user. 
startDatetime is An event, schedule, or schedule begins. Must specify "AM and PM"
endDatime is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "AM
location means meeting someone or a place. 
I hope the content includes anything other than behavioral and planning.
The answer must always be "JSON TYPE"
"""

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

def bd_params(input_text):
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
    
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        return {}  # JSON 디코딩에 실패하면 빈 딕셔너리 반환 또는 오류 처리
    
    # 바로 FastAPI 엔드포인트의 응답 형식으로 반환
    return ParameterExtractionResponse(
        title=response_json.get("title", ""),
        startDatetime=response_json.get("startDatetime", ""),
        endDatetime=response_json.get("endDatetime", ""),
        location=response_json.get("location", ""),
        content=response_json.get("content", "")
    )


# 요청과 응답을 위한 Pydantic 모델 정의
class InputText(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    classification: str

# Pydantic 모델 정의
class ParameterExtractionRequest(BaseModel):
    input_text: str

class ParameterExtractionResponse(BaseModel):
    title: str
    startDatetime: str
    endDatetime: str
    location: str
    content: str

class Goal(BaseModel):
    title: str
    start_date: str
    end_date: str
    location: str
    content: str

# Node.js 서버의 엔드포인트 URL
node_url = '43.201.211.135:3000/goal/summary2'  # 수정 필요

try:
    # Node.js 서버에 GET 요청 보내기
    response = requests.get(node_url)

    # 응답 코드 확인
    if response.status_code == 200:
        # 응답 데이터 출력
        print("목표 목록:")
        print(response.text)
    else:
        print(f"오류 응답 코드: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"오류 발생: {str(e)}")


# @app.post("43.201.211.135:3000/goal/summary", response_model=List[Goal])
# async def receive_summary_data(request_data: List[Goal]):
#     try:
#         # 목표 정보 목록을 반복하면서 OpenAI에 요청을 보내고 응답을 목표 정보에 추가합니다.
#         for goal_item in request_data:
#             # 각 목표 정보의 제목을 가져와서 OpenAI에 요청합니다.
#             input_text = goal_item.title

#             # OpenAI 요청을 보냅니다.
#             response = openai.Completion.create(
#                 engine="text-davinci-003",
#                 prompt=f"{prompt}\nUser: {input_text}\n",
#                 max_tokens=500,
#                 temperature=0.7,
#                 frequency_penalty=0.0,
#                 presence_penalty=0.0,
#                 stop=None
#             )

#             # OpenAI에서 받은 응답을 목표 정보에 추가합니다.
#             goal_item.summary = response.choices[0].text.strip()

#         # 수정된 목표 정보 목록을 반환합니다.
#         return request_data
#     except Exception as e:
#         # 예외 처리
#         raise HTTPException(status_code=500, detail="An error occurred while generating the response")



# # 요약 엔드포인트
# @app.post("/summarize")
# async def summarize(input_text: InputText):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=f"{prompt}\nUser: {input_text.text}\n",
#         max_tokens=500,
#         temperature=0.7,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#         stop=None
#     )
#     return {"summary": response.choices[0].text.strip()}

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

@app.get("/goal/summary")

# 파라미터 추출 엔드포인트 생성
@app.post("/extract_parameters", response_model=ParameterExtractionResponse)
async def extract_parameters(request_data: ParameterExtractionRequest):
    input_text = request_data.input_text
    extracted_parameters = bd_params(input_text)  # 파라미터 추출 로직을 호출하고 결과를 extracted_parameters에 할당

    # 추출된 파라미터를 ParameterExtractionResponse 모델 형식으로 변환하여 반환
    return ParameterExtractionResponse(
        title=extracted_parameters.get("title", ""),
        startDatetime=extracted_parameters.get("startDatetime", ""),
        endDatetime=extracted_parameters.get("endDatetime", ""),
        location=extracted_parameters.get("location", ""),
        content=extracted_parameters.get("content", "")
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
