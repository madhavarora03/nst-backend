import uvicorn
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from config import db
from routes import health_check, user

app = FastAPI()

print(db.name)

app.include_router(health_check.router, prefix="/api/health-check")
app.include_router(user.router, prefix="/api/user")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        reload=True,
    )
