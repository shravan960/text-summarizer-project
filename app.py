from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #ui
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# initialize our fastapi app
app=FastAPI(title="Text summarizer app", description="text summerization using T5", version="1.0")
# model and tokenizer

model=T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer=T5Tokenizer.from_pretrained("./saved_summary_model")


# device
if torch.backends.mps.is_available():
    device=torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
model.to(device)

# template
templates = Jinja2Templates(directory="./")

# inputy schema for dialogue=> string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text=re.sub("r\r\n", " ", text) # remove line
    text=re.sub(r"\s+", " ", text)# spaces
    text=re.sub("r<.*>", " ", text) # html tags
    text=text.strip().lower()
    return text


def summarize_dialogue(dialogue : str) -> str:
    dialogue=clean_data(dialogue) # clean

    #tokenizer
    inputs=tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    #generate the sumary=> token ids
    model.to(device)
    targets=model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # token ids convert to summary=> decoding

    summary=tokenizer.decode(targets[0], skip_special_tokens=True) #eos, sep

    return summary

# api endpoints

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary= summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/")
async def home():
    return FileResponse("index.html")
