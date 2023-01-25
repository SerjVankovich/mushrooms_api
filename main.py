from fastapi import FastAPI
from pydantic import BaseModel

import mushrooms_utils
import myshrooms_model


class Picture(BaseModel):
    data: str


app = FastAPI()

model = myshrooms_model.model


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/recognize/")
async def recognize_mushroom(picture: Picture):
    image = mushrooms_utils.decode_img(picture.data)
    prediction, probability = mushrooms_utils.predict(image, model)
    print(prediction, probability)
    return {
        "mushroomName": prediction,
        "probability": int(probability),
        "description": mushrooms_utils.DESCRIPTIONS[prediction]
    }


@app.get("/mushroom/sample/{mushroom_name}")
async def get_mushrooms_sample(mushroom_name: str, num: int = 10):
    return mushrooms_utils.get_mushrooms(mushroom_name, num)
