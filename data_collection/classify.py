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


PROMPT = """Analyze the given scientific text and provide classifications in the following order:

1. Synthesis Recipe Classification:
Determine if the text contains detailed synthesis procedures.
Return only "YES" or "NO".
If "NO", stop here. If "YES", continue with the following classifications.

2. Target Classification:
Classify the synthesized target as one of:
- Material (e.g., nanoparticles, compounds, composites)
- Device (e.g., sensors, batteries, transistors)
- Molecule (e.g., organic compounds, polymers)

3. Material Identification:
Provide:
- Chemical formula (if applicable)
- Material name
- Material class (e.g., metal oxide, polymer, semiconductor)

4. Application Domain:
List the primary applications mentioned in the text:
- Energy (e.g., batteries, solar cells)
- Electronics (e.g., transistors, sensors)
- Healthcare (e.g., drug delivery, imaging)
- Environmental (e.g., catalysis, filtration)
- Others (specify)

5. Synthesis Process Classification:
Classify the given synthesis method into one of these categories. If it combines multiple methods, label it as "Hybrid". If it doesn't fit any category, label it as "Others".

Categories:
1. Solid-State: solid-state reaction, ceramic method, sintering
2. Vapor Deposition: CVD, PVD, sputtering, evaporation
3. Mechanochemical: ball milling, mechanical alloying
4. Hydrothermal: solvothermal, pressurized solution
5. Pyrolysis: thermal decomposition, spray pyrolysis
6. Melt Quenching: rapid solidification, glass formation
7. Electrochemical: electrodeposition, anodization
8. Self-Assembly: molecular assembly, biomineralization
9. Solution-Based: precipitation, sol-gel, wet chemical synthesis
10. Biological: biomimetic, enzyme-mediated, microbial synthesis
11. Hybrid: combination of multiple methods
12. Others: novel or unconventional methods


Format the output as a structured list only if Step 1 is "YES".
For not available, use "N/A".
Do not provide explanations or additional commentary.

Example Output:
For a paper titled "Hydrothermal Synthesis of LiFePO4/C Composites for High-Performance Lithium-Ion Batteries":

1. Synthesis Recipe: YES
2. Target: Material
3. Material Identification:
- Chemical Formula: LiFePO4/C
- Material Name: Carbon-coated lithium iron phosphate
- Material Class: Phosphate composite
4. Application Domain: Energy (lithium-ion batteries)
5. Synthesis Process: Hydrothermal (solvothermal)"""



client = openai.OpenAI()


def classify_paper(md_file):
    with open(md_file, "r") as f:
        text = f.read()

    if len(text) < 100:
        return None
    if len(text) > 50000:
        text = text[:50000]

    message = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Scientific Paper:\n" + text},
        ],
        max_tokens=4096,
    )
    return message.choices[0].message.content

def main():
    md_dir = "../download_paper/markdowns"
    md_files = list(glob.glob(f"{md_dir}/*.md"))

    result_file = "classify-result-16K-4o-mini.jsonl"
    results = {}
    if os.path.exists(result_file):
        with jsonlines.open(result_file) as reader:
            for obj in reader:
                # results[obj["id"]] = obj["classification_result"]
                md_files.remove(f"{md_dir}/{obj['id']}.md")

    print("Markdown files:",len(md_files))

    def process_item(md_file: str):
        id = os.path.basename(md_file).replace(".md", "")
        try:
            result = classify_paper(md_file)
            if result:
                print(f"Processed {id}")
                return {"id": id, "classification_result": result}
            else:
                print(f"Failed to process {id}")
        except KeyboardInterrupt:
            print("Interrupted")
            raise
        except Exception as e:
            print(f"Error processing {id}: {e}")

    fout = jsonlines.open(result_file, "a")

    # for i, item in enumerate(tqdm(md_files)):
    #     result = process_item(item)
    #     if result:
    #         fout.write(result)

    
    lock = threading.Lock()
    num_threads = 8
    batch_size = 32

    def process_batch(batch):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_item, item) for item in batch]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        results = [result for result in results if result is not None]
        with lock:
            for item in results:
                fout.write(item)

    batch = []
    for i, item in enumerate(tqdm(md_files)):
        batch.append(item)
        if len(batch) == batch_size:
            process_batch(batch)
            batch = []
    if batch:
        process_batch(batch)

if __name__ == "__main__":
    main()

