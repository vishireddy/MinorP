from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import red, green, black, HexColor, gray
from pypdf import PdfReader, PdfWriter

IN_DIFF = 'changes.diff'
OUT_DIFF_PDF = 'changes_highlighted.pdf'
CORRECTED = 'LexPulse_ISMAC2026_corrected.pdf'
OUT_FINAL = 'LexPulse_ISMAC2026_annotated.pdf'

# Read diff
with open(IN_DIFF, 'r', encoding='utf-8') as f:
    lines = f.readlines()

c = canvas.Canvas(OUT_DIFF_PDF, pagesize=A4)
width, height = A4
left = 15 * mm
top = height - 20 * mm
y = top
line_height = 6.5 * mm
c.setFont('Courier', 9)

for raw in lines:
    line = raw.rstrip('\n')
    if y < 25 * mm:
        c.showPage()
        c.setFont('Courier', 9)
        y = top
    color = black
    if line.startswith('+++') or line.startswith('---') or line.startswith('diff') or line.startswith('index'):
        color = gray
    elif line.startswith('+'):
        color = green
    elif line.startswith('-'):
        color = red
    elif line.startswith('@'):
        color = HexColor('#ff8800')
    else:
        color = black
    c.setFillColor(color)
    # truncate if too long
    max_chars = 120
    text = line
    if len(text) > max_chars:
        text = text[:max_chars-3] + '...'
    c.drawString(left, y, text)
    y -= line_height

c.save()

# Merge highlighted diff PDF with corrected PDF
writer = PdfWriter()
reader_diff = PdfReader(OUT_DIFF_PDF)
for p in reader_diff.pages:
    writer.add_page(p)
reader_corr = PdfReader(CORRECTED)
for p in reader_corr.pages:
    writer.add_page(p)
with open(OUT_FINAL, 'wb') as f:
    writer.write(f)
print('Created', OUT_FINAL)
