import os
from fpdf import FPDF

def create_pdf(text_file, pdf_file):
    if not os.path.exists(text_file):
        return
        
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
        # Fix unicode issues for fpdf
        text = text.replace("—", "-").replace("–", "-")
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text=text)
    pdf.output(pdf_file)
    print(f"Created {pdf_file}")

if __name__ == "__main__":
    create_pdf("data/raw/base_policy.txt", "data/raw/base_policy.pdf")
    create_pdf("data/raw/amendment_policy.txt", "data/raw/amendment_policy.pdf")
    # Clean up txt files
    if os.path.exists("data/raw/base_policy.txt"): os.remove("data/raw/base_policy.txt")
    if os.path.exists("data/raw/amendment_policy.txt"): os.remove("data/raw/amendment_policy.txt")
