from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Define prompt
prompt = """Bot: How can I help you?
User: {{}}
---------------------------------------------
You are a schedule summary. Displays a summary of your schedule in the latest order based on today's date.
"""

# summarize 기능 함수
def get_summarize():
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt}",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()

# Node.js 서버의 엔드포인트 URL
node_url = 'http://43.201.211.135:3000/goal/summary'  # 수정 필요

class BotResponse(BaseModel):
    response: str

@app.get("/get_bot_response", response_model=BotResponse)
async def get_bot_response():
    try:
        # Node.js 서버에 GET 요청 보내기
        response = requests.get(node_url)

        # 응답 코드 확인
        if response.status_code == 200:
            # 응답 데이터 출력
            print("목표 목록:")
            print(response.text)

            # 챗봇 응답 생성 (Node.js 서버에서 받은 데이터를 그대로 사용)
            bot_response = response.text
            return {"response": bot_response}
        else:
            error_message = f"오류 응답 코드: {response.status_code}\n{response.text}"
            raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"오류 발생: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# @app.post("/chat_summarize", response_model=):

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
