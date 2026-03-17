from fastapi import FastAPI
from sqlalchemy import create_engine
from app.models.experiment import Base

DATABASE_URL = "sqlite:///./capstone.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# DB 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Capstone API")

@app.get("/")
def root():
    return {"status": "ok"}