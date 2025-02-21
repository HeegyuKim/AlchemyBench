import jsonlines, json
import openai
from tqdm import tqdm
import time
from pprint import pprint
import time

def process_batch(client, model: str, id_list, messages_list, job_description="paper extraction job", body_kwargs={}):
    temp_file = f"temp_{model}_{time.time()}.jsonl"
    print("Temp File:", temp_file)
    with jsonlines.open(temp_file, "w") as fout:
        for item_id, messages in zip(id_list, messages_list):
            # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
            # {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
            fout.write({
                "custom_id": item_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    **body_kwargs
                }
            })

    batch_input_file = client.files.create(
        file=open(temp_file, "rb"),
        purpose="batch"
    )
    print("Batch Input File:", batch_input_file)

    batch_input_file_id = batch_input_file.id
    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": job_description
        }
    )

    batch_id = batch_obj.id
    waiting_status = ["in_progress", "finalizing", "validating"]
    i = 0
    while True:
        batch = client.batches.retrieve(batch_id)
        print("Batch Object:", batch)
        status = batch.status
        if status in waiting_status:
            print("Batch Status:", status, "Waiting for 60 seconds")
            time.sleep(60)
        elif status == "completed":
            break
        else:
            pprint(batch)
            raise Exception(f"Batch failed: {status}")
            

    pprint(batch)
    results = retrieve_batch_responses(client, batch.output_file_id)
    client.files.delete(batch_input_file_id)
    client.files.delete(batch.output_file_id)
    return results

def retrieve_batch_responses(client, batch_id: str):
    file_response = client.files.content(batch_id)
    # {"id": "batch_req_123", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_123", "body": {"id": "chatcmpl-123", "object": "chat.completion", "created": 1711652795, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}, "system_fingerprint": "fp_123"}}, "error": null}
    # {"id": "batch_req_456", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "req_789", "body": {"id": "chatcmpl-abc", "object": "chat.completion", "created": 1711652789, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?"}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}, "system_fingerprint": "fp_3ba"}}, "error": null}
    responses = file_response.text.split("\n")
    responses = [json.loads(item) for item in responses if item]

    outputs = {x["custom_id"]: x["response"]["body"]["choices"][0]["message"]["content"] for x in responses}

    return outputs
