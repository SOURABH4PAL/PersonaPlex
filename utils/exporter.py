from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import os


def generate_pdf(text: str):

    os.makedirs("exports", exist_ok=True)
    file_path = "exports/personaplex_result.pdf"

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontSize = 11
    style.leading = 16
    style.alignment = TA_LEFT

    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line.replace("&", "&amp;"), style))
        story.append(Spacer(1, 8))

    doc.build(story)

    return file_path


def generate_csv(text: str):
    import csv

    os.makedirs("exports", exist_ok=True)
    file_path = "exports/personaplex_result.csv"

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PersonaPlex Output"])

        for line in text.split("\n"):
            writer.writerow([line])

    return file_path

