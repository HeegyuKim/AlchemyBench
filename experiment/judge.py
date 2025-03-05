import os
from litellm import batch_completion, completion
import openai
import fire
from datasets import load_dataset
from pprint import pprint
import jsonlines
from tqdm import tqdm
import openai_utils
from predict import RecipePredictor


USER_PROMPT = """
Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}

Ground Truth Recipe:
{gt_recipe}"""

class RecipeJudge(RecipePredictor):

    def __init__(self, model="gpt-4o-mini", batch_size=1, max_tokens=4096, max_completion_tokens=16384, temperature=0, api_key=None, prompt_filename = "prompts/prediction_0209.txt"):
        super().__init__(model, batch_size, max_tokens, max_completion_tokens, temperature, api_key, prompt_filename)
        self.job_description = f"material LLM judge job w/ {model}"

    def build_prompt(self, item):
        contributions, recipe = item["contribution"], item["recipe"]
        system_prompt = self.prediction_prompt
        return [
            {
                "content": system_prompt,
                "role": "system"
            },
            {
                "content": USER_PROMPT.format(objective=contributions, prediction=item["prediction"], gt_recipe=recipe),
                "role": "user"
            }
        ]



def main(
        filename: str,
        model: str = "gpt-4o-2024-08-06",
        prompt_name: str = "judge",
        batch_size: int = 1,
        use_openai_batch: bool = False,
):
    ds = list(jsonlines.open(filename))
    prompt_filename = f"prompts/{prompt_name}.txt"
    predictor = RecipeJudge(model=model, prompt_filename=prompt_filename)
    model_name = model.split("/", 1)[-1]
    output_filename = filename.replace(".jsonl", f"_judged_{model_name}.jsonl")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        ds = ds[skip:]
        print(f"Skipping {skip} items")

    with jsonlines.open(output_filename, "a") as fout:
        for item, prediction in tqdm(predictor.predict(ds, batch_size=batch_size, use_openai_batch=use_openai_batch), total=len(ds)):
            item["judge_result"] = prediction
            item["judge_model"] = model
            fout.write(item)
            # print(item)
            # print(prediction)

if __name__ == "__main__":
    fire.Fire(main)