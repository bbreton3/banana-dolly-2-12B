from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer
from instruct_pipeline import InstructionTextGenerationPipeline

import torch

app = Potassium("my_app")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    MODEL_NAME = "databricks/dolly-v2-2-8b"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    context = {"pipeline": pipeline}

    return context


# @app.handler is an http post handler running for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    pipeline = context.get("pipeline")

    # Run the model
    result = pipeline(prompt)

    return Response(json={"outputs": result}, status=200)


if __name__ == "__main__":
    app.serve()
