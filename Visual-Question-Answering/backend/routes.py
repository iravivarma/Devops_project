''' @InProceedings{pmlr-v139-kim21k, title = 	 {ViLT: Vision-and-Language
Transformer Without Convolution or Region Supervision}, author =       {Kim,
Wonjae and Son, Bokyung and Kim, Ildoo}, booktitle = 	 {Proceedings of the
38th International Conference on Machine Learning}, pages = 	 {5583--5594},
year = 	 {2021}, editor = 	 {Meila, Marina and Zhang, Tong}, volume = 	 {139},
series = 	 {Proceedings of Machine Learning Research}, month = 	 {18--24 Jul},
publisher =    {PMLR}, pdf = 	
{http://proceedings.mlr.press/v139/kim21k/kim21k.pdf}, url = 	
{http://proceedings.mlr.press/v139/kim21k.html}, abstract = 	
{Vision-and-Language Pre-training (VLP) has improved performance on various
joint vision-and-language downstream tasks. Current approaches to VLP heavily
rely on image feature extraction processes, most of which involve region
supervision (e.g., object detection) and the convolutional architecture (e.g.,
ResNet). Although disregarded in the literature, we find it problematic in
terms of both (1) efficiency/speed, that simply extracting input features
requires much more computation than the multimodal interaction steps; and (2)
expressive power, as it is upper bounded to the expressive power of the visual
embedder and its predefined visual vocabulary. In this paper, we present a
minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the
sense that the processing of visual inputs is drastically simplified to just
the same convolution-free manner that we process textual inputs. We show that
ViLT is up to tens of times faster than previous VLP models, yet with
competitive or better downstream task performance. Our code and pre-trained
weights are available at https://github.com/dandelin/vilt.} } '''


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

#if __name__ == "__main__":
#    uvicorn.run("route:app", host="0.0.0.0", port=8080)








