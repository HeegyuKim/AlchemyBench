# Supplementary Materials
Use python=3.11 and install the required packages using the following command:
```
pip install -r requirements.txt
```

## Repository Structure
```
data_collection/
- Code for retrieving articles from Semantic Scholar API (retrieve_s2api.py)
- Download the PDFs of the articles (download_paper.py)
- Code for converting PDFs to markdowns (pdf2md.py)
- Code for classifying and extracting recipes from markdowns (classify.py and extract_recipe.py)

experiment/
- Code for predicting recipes using the Open Materials Guide (OMG) dataset (predict_recipe.py)
- Code for evaluating the performance of the model using LLM-as-a-Judge (evaluate.py)
```
