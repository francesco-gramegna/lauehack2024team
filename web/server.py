from typing import Union
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()


app.mount("/", StaticFiles(directory="static"), name="static")


class Request(BaseModel):
    var: str
    amount: float
    time: int
    duration: int
    actionType: str



@app.post("/forecast/")
async def create_item(req: Request):
    print(req)

    return { "a": "a" }