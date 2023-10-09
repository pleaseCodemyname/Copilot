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

load_dotenv()
app = FastAPI()


api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

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

# Pydantic 모델 정의
class ParameterExtractionRequest(BaseModel):
    input_text: str

# 파라미터 추출 엔드포인트 생성
@app.post("/create/extract_parameters")
async def extract_parameters(request_data: ParameterExtractionRequest):  # Request 매개변수 추가
    input_text = request_data.input_text
    extracted_parameters = bd_params(input_text)  # 파라미터 추출 로직을 호출하고 결과를 extracted_parameters에 할당
    return extracted_parameters

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)