from fastapi import FastAPI
from app.routers import health_check, prediction

app = FastAPI()

app.include_router(health_check.router)
app.include_router(prediction.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)