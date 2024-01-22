from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from cv_sender import respond_unread_emails

app = FastAPI()


@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 24)  # repeat every 24h
def initialization():
    print("executing periodic reading of emails")
    respond_unread_emails()


@app.get("/")
async def root():
    return {"message": "Resume sending API is up and running!"}