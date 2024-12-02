import fitz  # PyMuPDF
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def process_and_save_data(pdf_folder, output_folder):
    """
    Extract text from PDFs in the given folder and save as Parquet files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            print(f"Extracted text from {pdf_path}")
            data = [{"file_name": pdf_file, "content": content.strip()}
                    for content in text.split("\n") if content.strip()]
            df = pd.DataFrame(data)
            parquet_file = os.path.splitext(pdf_file)[0] + ".parquet"
            df.to_parquet(os.path.join(output_folder, parquet_file))

if __name__ == "__main__":
    process_and_save_data("pdfs", "dataset")
