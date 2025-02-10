from dotenv import load_dotenv

load_dotenv()

from config import db
from fastapi import FastAPI
from routes import health_check

app = FastAPI()

print(db.name)

app.include_router(health_check.router, prefix="/health-check")
