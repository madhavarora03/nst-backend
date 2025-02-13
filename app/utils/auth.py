import os
from datetime import datetime, timedelta

import jwt
from fastapi import Request, HTTPException
from passlib.context import CryptContext
from pydantic import BaseModel


# User schema for token response
class TokenData(BaseModel):
    email: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key & algorithm for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# Function to hash password
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# Function to verify passwords
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# Function to create JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Function to verify JWT token
def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return TokenData(email=email)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Dependency to get current user
def get_current_user(request: Request):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return verify_access_token(token)
