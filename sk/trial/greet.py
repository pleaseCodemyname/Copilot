from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI
import uvicorn

oauth_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

def get_current_user(token: str = Depends(oauth_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception

# 환영 메시지 엔드포인트
@app.get("/welcome")
async def welcome(user_id: str = Depends(get_current_user)):
    return {"message": f"안녕하세요 {user_id}님. 무엇을 도와드릴까요?"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
