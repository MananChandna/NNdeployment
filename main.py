from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import src.pipeline as pl

from src.preprocessing.data_management import load_model
from src.predict import inference

saved_file_name = "two_input_xor_nn"
loaded_model = load_model(saved_file_name)

app = FastAPI(
    title="Two Input XOR Function Implementor",
    description="A Two Input Neural Network to implement XOR Function",
    version="0.1"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TwoInputXORGate(BaseModel):
    X1: float
    X2: float

@app.get("/")
def index():
    return {'message': 'A Web App for serving the output of two input XOR Function implemented through neural network'}

@app.post("/generate_response")
def generate_response(trigger: TwoInputXORGate):
    input1 = trigger.X1
    input2 = trigger.X2

    input_to_nn = np.array([input1, input2])
    nn_out = inference(input_to_nn, loaded_model["params"]["biases"], loaded_model["params"]["weights"])

    return {"output": nn_out.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)