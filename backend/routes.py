from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from typing import Optional

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import io, base64


#model_path="C:\Users\ravivarmainjeti\Desktop\projects\visual_question_answering\vilt-b32-finetuned-vqa\pytorch_model.bin"
app = FastAPI()


def get_answer(content, question):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(content, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])

    return model.config.id2label[idx]


@app.post("/inputs")
def ask_question(file_: UploadFile = File(None), image_bytes: str = Form(None),  question:str = Form(...)):
    # print(type(file_))
    if file_:
        content = file_.file.read()
    else: 
        print(type(image_bytes))
        content=base64.b64decode(image_bytes.encode('utf-8'))
        print(type(content))
    image = Image.open(io.BytesIO(content))
    print(type(image))
    response = get_answer(image, question)

    return response

if __name__ == "__main__":
    uvicorn.run("route:app", host="0.0.0.0", port=8080)








