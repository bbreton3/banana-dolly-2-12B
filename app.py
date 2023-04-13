from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer
from instruct_pipeline import InstructionTextGenerationPipeline

import torch
import time

app = Potassium("my_app")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    MODEL_NAME = "databricks/dolly-v2-12b"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    context = {"model": model, "tokenizer": tokenizer}

    return context


# @app.handler is an http post handler running for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    # Start timer
    t_1 = time.time()

    prompt = request.json.get("prompt")

    do_sample = request.json.get("do_sample", True)
    max_new_tokens = request.json.get("max_new_tokens", 256)
    top_p = request.json.get("top_p", 0.92)
    top_k = request.json.get("top_k", 0)

    # Create pipeline with the following parameters
    pipeline = InstructionTextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        do_sample=do_sample,
        max_new_tokens=int(max_new_tokens),
        top_p=float(top_p),
        top_k=int(top_k),
    )
    # Run the model
    result = pipeline(prompt)

    t_2 = time.time()

    return Response(
        json={
            "output": result,
            "prompt": prompt,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "inference_time": t_2 - t_1,
        },
        status=200,
    )


if __name__ == "__main__":
    app.serve()
