from fastapi import FastAPI, HTTPException
import requests
import uvicorn

app = FastAPI()
BASE_URL = 'http://43.201.211.135:3000'
session = None  # 세션을 전역으로 둡니다.

@app.on_event("startup")
async def startup_event():
    # 애플리케이션이 시작될 때 세션을 초기화합니다.
    global session
    session = requests.Session()

@app.post("/login")
async def login(user_id: str, password: str):
    login_payload = {'user_id': user_id, 'password': password}
    login_response = requests.post(f'{BASE_URL}/account/login', json=login_payload)

    if login_response.status_code == 200:
        session.headers.update({'Authorization': f'Bearer {login_response.json()["token"]}'})
        return {"status": "success"}
    else:
        raise HTTPException(status_code=login_response.status_code, detail=login_response.json())

@app.get("/goals")
async def get_goals():
    goals_response = session.get(f'{BASE_URL}/goal/read')

    if goals_response.status_code == 200:
        goals = goals_response.json()
        return {"goals": goals}
    else:
        raise HTTPException(status_code=goals_response.status_code, detail=goals_response.json())


