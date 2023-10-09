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
import requests
from typing import Optional
from time import sleep
# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

prompt4 = """Bot: How can I help you?
User: {{$input}}
If the user says that the schedule on the date or time is canceled or gone, I will check the schedule on that date.
And please keep asking until you extract the exact schedule according to the user's question, 
ask "Do you want to delete it" if the schedule fits, and answer "Yes" and "No" at the end.
"""

class InputText(BaseModel):
    text: str

@app.post("/delete_plan")
async def del_plan(input_text: InputText):
    #Node의 데이터 가져오기
    node_data = get_data_from_node(input_text.text)

    #process the data as needed
    result = del_plan_logic(node_data)
    return result

def get_data_from_node(input_text):
    # Node.js 서버의 API 엔드포인트 (event/read)
    node_api_endpoint = "http://43.201.211.135:3000/event/read"

     # HTTP POST 요청 보내기
    response = requests.post(node_api_endpoint, json={"input_text": input_text})

    # 출력을 통해 응답 확인
    print(response)

    try:
        # JSON 형식의 응답 데이터 추출
        data = response.json()
    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코드 오류: {e}")
        data = None

    return data
# 입력을 분류하는 함수
def del_plan_logic(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt4}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()

# FastAPI 애플리케이션을 실행합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)