from fastapi import FastAPI, Request

# BaseModel is used to define the structure of the data your API expects.
# FastAPI uses Pydantic behind the scenes to validate request data automatically.
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App",description="Text Summarization using T5",version="1.0")

# model & tokenizer
MODEL_NAME = "Tisha34/Abstractive-Text-Summarizer-T5-Transformers"

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".") # current dir

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue : str    #defines a field

def clean_data(text):
  text = re.sub(r"\r\n"," ",text) # lines
  text = re.sub(r"\s+"," ",text) # spaces
  text = re.sub(r"<.*?>"," ",text) # html tags
  text = text.strip().lower()
  return text

def summarize_dialogue(dialogue : str):
    dialogue = "summarize: " + clean_data(dialogue) # clean

    # tokenize dialogue
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt").to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(input_ids = inputs["input_ids"],
                            attention_mask = inputs["attention_mask"],
                            max_length=150,
                            min_length=20,
                            length_penalty=2.5,
                            num_beams = 6, # Think of 4 possible summaries and choose the best one.
                            no_repeat_ngram_size=3,
                            early_stopping=True,)
    
    # convert targets token ids into summary -> decoding
    summary = tokenizer.decode(targets[0],skip_special_tokens=True)
    return summary


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
   return templates.TemplateResponse(request=request,
        name="index.html",context={})

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
   summary = summarize_dialogue(dialogue_input.dialogue)
   return {"summary":summary}
