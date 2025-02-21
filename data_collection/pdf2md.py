import os
import multiprocessing as mp
from pathlib import Path
import pymupdf4llm

def convert_pdf_to_markdown(pdf_path):
    try:
        output_path = pdf_path.with_suffix('.md')
        output_path = Path("markdowns") / output_path.name
        if output_path.exists():
            return
        
        md_text = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
        output_path.write_bytes(md_text.encode())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        pass

def process_directory(input_dir):
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    with mp.Pool(processes=8) as pool:
        pool.map(convert_pdf_to_markdown, pdf_files)

if __name__ == "__main__":
    input_directory = "./pdfs"
    process_directory(input_directory)
