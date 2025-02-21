import requests
import time
import os
import jsonlines
import time
from tqdm import tqdm

DELAY_TIME = 360

def search_papers_with_pdf(query, year: str = None, max_results=1000, token=None):
    base_url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

    num_gets = 0
    progress = tqdm(total=max_results, desc=f"Searching: {query}")

    while num_gets < max_results:
        params = {
            'query': query,
            'fields': 'title,year,authors,abstract,url,openAccessPdf,venue,citationCount,externalIds,publicationVenue',
            'pdf': True,
            'fieldsOfStudy': 'Chemistry,Materials Science'
        }
        if year:
            params['year'] = year
        if token:
            params['token'] = token
        
        headers = {
            'Accept': 'application/json',
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        total = data.get('total', 0)

        papers_with_pdf = [
            paper for paper in data.get('data', [])
        ]
        
        if not papers_with_pdf:
            break
            
        yield from papers_with_pdf
        
        num_gets += len(papers_with_pdf)
        token = data.get('token')
        progress.total = total
        progress.update(len(papers_with_pdf))

        if num_gets >= max_results or token is None:
            break

        time.sleep(DELAY_TIME)
            
    
    
queries = [
    # Solid-State Processing
    "solid state sintering process", "reactive sintering synthesis",
    "pressure-assisted sintering", "spark plasma sintering",
    "hot pressing synthesis", "hot isostatic pressing",
    "cold isostatic pressing", "flash sintering technique",
    "field-assisted sintering", "microwave sintering process",

    # Mechanochemical Methods
    "high energy ball milling", "mechanical alloying synthesis",
    "mechanochemical activation", "planetary ball milling",
    "cryogenic milling process", "attrition milling synthesis",
    "mechanical grinding method", "mechanofusion process",
    "mechano-chemical reaction", "solid-state mechanical synthesis",

    # Vapor Deposition Techniques 
    "atomic layer deposition", "plasma enhanced CVD",
    "metal organic CVD", "low pressure CVD",
    "atmospheric pressure CVD", "electron beam PVD",
    "magnetron sputtering deposition", "pulsed laser deposition",
    "thermal evaporation method", "molecular beam epitaxy",

    # Advanced Thermal Methods
    "combustion synthesis process", "self-propagating synthesis",
    "plasma spray synthesis", "flame spray pyrolysis",
    "laser ablation synthesis", "thermal plasma synthesis",
    "microwave-assisted synthesis", "ultrasonic spray pyrolysis",
    "radio frequency thermal plasma", "arc discharge synthesis",

    # Electrochemical Approaches
    "electrochemical co-deposition", "pulse electrodeposition",
    "electroless deposition method", "anodic oxidation synthesis",
    "cathodic reduction process", "electrochemical etching",
    "electrochemical polymerization", "electrophoretic deposition",
    "galvanic replacement reaction", "electrochemical exfoliation",

    # Novel Processing Methods
    "freeze drying synthesis", "spray freeze drying",
    "supercritical fluid process", "template-assisted synthesis",
    "biomimetic processing method", "sol-gel electrospinning",
    "ionothermal synthesis route", "microemulsion technique",
    "sonochemical processing", "continuous flow synthesis"
]

year = None
max_results = 100000

for query in queries:
    fout = jsonlines.open(f"s2api-result/bulk-papers-{query.replace(' ', '_')}-{year}.jsonl", "w")
    for i, paper in enumerate(search_papers_with_pdf(query, max_results=max_results, year=year)):
        fout.write(paper)
    
    time.sleep(DELAY_TIME)