import multiprocessing as mp
from functools import partial
import openai
import pymupdf4llm
import glob
import os
import jsonlines
from tqdm import tqdm
import threading, concurrent.futures
import json
import pandas as pd
import fire


PROMPT = """You are a materials science expert. Your task is to extract ONLY the explicitly stated synthesis information from the provided research paper. Do not generate, assume, or infer any information not directly presented in the paper.
If the provided paper does not contain any synthesis information, please indicate "NOT A MATERIAL SYNTHESIS PAPER" and do not provide any further details.

## Key Contributions
Summarize the key contributions of the paper:
- Novel materials or compounds: <summary>
- Unique synthesis methods: <summary>
- Specific applications or domains: <summary>

## Materials
Extract and list:
- All precursor materials with:
  * Exact quantities and concentrations
  * Molar ratios or stoichiometric proportions
  * Purity grades and specifications
  * Supplier information if provided
- Solvents, reagents, catalysts, and any other materials such as carrier gases.

## Synthesis Equipment
- All equipment and apparatus with:
  * Model numbers if specified
  * Operating parameters
  * Special configurations or modifications

## Synthesis Procedure
Extract and organize:
- Chronological step-by-step synthesis method
- All processing parameters:
  * Temperature ranges and ramp rates
  * Time durations for each step
  * Pressure conditions
  * pH values if applicable
  * Mixing speeds and durations
- Critical control points and special conditions

## Characterization Methods and Equipment
List all:
- Analytical techniques used
- Specific measurement conditions
- Sample preparation methods
- Equipment models and settings
- Standards or references used

## Product Characteristics
Document:
- Final product properties and specifications (include both numerical values and literal descriptions if provided)
- Yield calculations and actual yields
- Purity levels and impurity content
- Performance metrics with measured values
- Morphological characteristics

IMPORTANT RULES:
1. DO NOT generate or assume any missing information
2. If specific details are not mentioned in the paper, indicate "N/A"
3. Use exact numbers and units as presented in the paper
4. Maintain original measurement units
5. Quote unusual or specific procedures directly when necessary
6. Format all information using proper markdown with headers (##) and bullet points

Remember: Accuracy and authenticity are crucial. Only include information explicitly stated in the paper."""

client = openai.OpenAI()



def make(
    input_file: str,
    batch_size: int = 1024,
    model: str = "gpt-4o",
):
    result_file = input_file.replace(".jsonl", f"-recipe-{model}.jsonl")
    df = pd.read_json(input_file, lines=True)

    def make_body(item):
        md_file = f"../download_paper/markdowns/{item.id}.md"

        with open(md_file, "r") as fin:
            text = fin.read()

        if len(text) < 100:
            return None
        if len(text) > 50000:
            text = text[:50000]

        body = dict(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "Scientific Paper:\n" + text},
            ],
            max_tokens=4096,
        )
        return body
    
    df = df[df.classification_result.apply(lambda x: "target: material" in x.lower() if isinstance(x, str) else False)]
    # df = df[df.year.apply(lambda x: int(x) >= 2024 if x != "N/A" else False)]

    if os.path.exists(result_file):
        results = list(jsonlines.open(result_file))
        results = {item["id"]: item for item in results}
    else:
        results = {}
    
    df = df[~df.id.isin(results.keys())]
    print(f"Total papers: {df.shape[0]}")
    print(df.head())

    fout = jsonlines.open(result_file, "a")

    rows = df.iterrows()

    def process_batch(batch):
        with jsonlines.open("batch_request.jsonl", "w") as fout:
            for item in batch:
                body = make_body(item)
                # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
                # {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
                fout.write({
                    "custom_id": item.id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                })

        batch_input_file = client.files.create(
            file=open("batch_request.jsonl", "rb"),
            purpose="batch"
        )
        print("Batch Input File:", batch_input_file)

        batch_input_file_id = batch_input_file.id
        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "paper extraction job"
            }
        )
        print("Batch Object:", batch_obj)

    batch = []
    for i, item in tqdm(rows, total=len(df)):
        batch.append(item)
        if len(batch) == batch_size:
            process_batch(batch)
            batch = []
            # break

    if batch:
        process_batch(batch)

    fout.close()

def get(batch_id: str,
        input_file: str,
        model: str = "gpt-4o"
        ):
    client = openai.OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(batch)


    file_response = client.files.content(batch.output_file_id)
    # {"id": "batch_req_123", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_123", "body": {"id": "chatcmpl-123", "object": "chat.completion", "created": 1711652795, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}, "system_fingerprint": "fp_123"}}, "error": null}
    # {"id": "batch_req_456", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "req_789", "body": {"id": "chatcmpl-abc", "object": "chat.completion", "created": 1711652789, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?"}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}, "system_fingerprint": "fp_3ba"}}, "error": null}
    responses = file_response.text.split("\n")
    responses = [json.loads(item) for item in responses if item]

    results = list(jsonlines.open(input_file))
    results = {item["id"]: item for item in results}
    
    result_file = input_file.replace(".jsonl", f"-recipe-{model}.jsonl")
    print("Total Responses:", len(responses))

    for response in tqdm(responses):
        item = results[response["custom_id"]]
        item["recipe"] = response["response"]["body"]["choices"][0]["message"]["content"]
        # print(item["id"], item["classification_result"], item["recipe"])
        # break
        with jsonlines.open(result_file, "a") as fout:
            fout.write(item)

if __name__ == "__main__":
    fire.Fire({
        "make": make,
        "get": get
    })
