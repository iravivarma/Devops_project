'''
bibtex @article{DBLP:journals/corr/abs-1910-13461, author = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad
 and Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov and Luke Zettlemoyer}, title = {{BART:} Denoising Sequence-to-Sequence
 Pre-training for Natural Language Generation, Translation, and Comprehension}, journal = {CoRR}, volume = {abs/1910.13461}, year = {2019},
 url = {http://arxiv.org/abs/1910.13461}, eprinttype = {arXiv}, eprint = {1910.13461}, timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},  
 biburl = {https://dblp.org/rec/journals/corr/abs-1910-13461.bib}, bibsource = {dblp computer science bibliography, https://dblp.org} }
'''

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from typing import Optional
from schemas import input_params, OutputResponse, OutputSchema
from transformers import pipeline
import requests

app=FastAPI()
def summarize(text, min_len, max_len):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_text=summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)
    print(summary_text)
    return summary_text

@app.post("/input_string", response_model=OutputResponse)
def text_summarization(inputs: input_params):

    countOfWords = len(inputs.text.split())

    if countOfWords < inputs.min_length or countOfWords > inputs.max_length:
        return(f"The string should be between {inputs.min_length} and {inputs.max_length}")
    else:
        results = summarize(inputs.text, inputs.min_length, inputs.max_length)[0]["summary_text"]
        results = results if isinstance(results, list) else [results] 
        return OutputResponse(
            data = [OutputSchema(text=result) for result in results]
        )
