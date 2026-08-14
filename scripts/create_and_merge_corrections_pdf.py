from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm
from pypdf import PdfReader, PdfWriter

import textwrap

MD_IN = 'response_to_reviewers_full.md'
CORRECTION_PDF = 'corrections.pdf'
ORIG_PDF = 'LexPulse_ISMAC2026.pdf'
OUT_PDF = 'LexPulse_ISMAC2026_corrected.pdf'

# Read markdown text
with open(MD_IN, 'r', encoding='utf-8') as f:
    md = f.read()

# Simple markdown -> paragraphs splitter (naive)
blocks = []
for line in md.split('\n'):
    if line.strip() == '':
        blocks.append('\n')
    else:
        blocks.append(line)

styles = getSampleStyleSheet()
body = styles['BodyText']
body.leading = 14
body.fontSize = 10

# Create corrections PDF
doc = SimpleDocTemplate(CORRECTION_PDF, pagesize=A4,
                        rightMargin=20*mm, leftMargin=20*mm,
                        topMargin=20*mm, bottomMargin=20*mm)
story = []

# Title
title_style = styles['Title']
title_style.fontSize = 16
story.append(Paragraph('LexPulse — Corrections & Reviewer Response', title_style))
story.append(Spacer(1, 6))

for para in md.split('\n\n'):
    text = para.strip()
    if not text:
        story.append(Spacer(1,6))
        continue
    # naive: convert markdown headers
    if text.startswith('#'):
        level = text.count('#', 0, text.find(' '))
        txt = text.strip('# ').strip()
        style = styles['Heading1'] if level==1 else styles['Heading2']
        story.append(Paragraph(txt, style))
        story.append(Spacer(1,4))
    else:
        # wrap long lines
        wrapped = '<br/>'.join(textwrap.wrap(text, 140))
        story.append(Paragraph(wrapped, body))
        story.append(Spacer(1,4))

try:
    doc.build(story)
except Exception as e:
    print('PDF build failed:', e)
    raise

# Merge corrections.pdf + original pdf
writer = PdfWriter()

# Add corrections first
reader_corr = PdfReader(CORRECTION_PDF)
for p in reader_corr.pages:
    writer.add_page(p)

# Append original
reader_orig = PdfReader(ORIG_PDF)
for p in reader_orig.pages:
    writer.add_page(p)

with open(OUT_PDF, 'wb') as f:
    writer.write(f)

print('Created', OUT_PDF)
