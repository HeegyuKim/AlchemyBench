import pymupdf4llm
from litellm import completion, batch_completion


PROMPT = """You are a materials science expert. Your task is to extract ONLY the explicitly stated synthesis information from the provided research paper. Do not generate, assume, or infer any information not directly presented in the paper.

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


import fitz  # PyMuPDF

def pdf_bytes_to_markdown(pdf_bytes):
    # Create a PyMuPDF document from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Convert to markdown
    md_text = pymupdf4llm.to_markdown(pdf_document)
    
    # Close the document to free resources
    pdf_document.close()
    
    return md_text

def extract_recipe_from_text(texts, model="gpt-4o-2024-11-20"):
    def filter_text(text):
        if len(text) < 100:
            return None
        if len(text) > 50000:
            text = text[:50000]
        return text
    
    texts = [filter_text(text) for text in texts]
    messages = [[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "Scientific Paper:\n" + text},
    ] for text in texts if text is not None]

    messages = batch_completion(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.6,
    )
    return [message.choices[0].message.content for message in messages]

def read_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
        text = pdf_bytes_to_markdown(pdf_bytes)
        return text
    
def pdf_bytelist_to_recipes(pdf_bytelist, model="gpt-4o-2024-11-20"):
    texts = [pdf_bytes_to_markdown(pdf_bytes) for pdf_bytes in pdf_bytelist]
    return extract_recipe_from_text(texts, model=model)

if __name__ == "__main__":
    pdf_files = ["test.pdf", "test.pdf"]
    texts = [read_pdf(pdf_file) for pdf_file in pdf_files]
    texts = extract_recipe_from_text(texts, model="gpt-4o-mini")
    print("\n\nExtracted Recipes:\n")
    for i, text in enumerate(texts):
        print(f"Recipe {i + 1}:\n{text}\n\n")