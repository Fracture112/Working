
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from datetime import datetime
from openai import OpenAI
import matplotlib.pyplot as plt
import re
import csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from streamlit_drawable_canvas import st_canvas

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_next_case_id():
    base = "cases"
    os.makedirs(base, exist_ok=True)
    existing = sorted([d for d in os.listdir(base) if d.startswith("case_")])
    if not existing:
        return os.path.join(base, "case_001")
    last = existing[-1]
    num = int(last.split("_")[-1]) + 1
    return os.path.join(base, f"case_{num:03d}")

def generate_gpt_analysis(description):
    prompt = f"""
You are a metallurgical failure analyst.

Analyze this fracture surface. Focus on:
- Beach marks → classify as fatigue only if visible
- Chevron marks → classify as bending only if present
- If no beach marks or origin is unclear → classify as overload
- Provide percentage confidence for: Bending, Torsional, Tensile, Reversed Bending, Rotating Bending

Image description:
{description}

Return a summary and this table:

| Feature | Analysis |
|--------|----------|
| Failure Mode | Fatigue / Overload |
| Type of Stress | Most likely |
| Beach Marks | Present / Not visible |
| Chevron Marks | Present / Absent |
| Origin Count | 0 / 1 / 2+ |
| Additional Notes | Surface indicators |

Add:
Bending: 40%
Torsional: 30%
Tensile: 20%
Reversed Bending: 5%
Rotating Bending: 5%
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional fracture analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content

def extract_fracture_description(image):
    np_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    desc = "This is a metal fracture surface. "
    if np.mean(edges) > 20:
        desc += "Surface appears rough. "
    else:
        desc += "Surface appears smooth. "
    return edges, desc

def extract_stress_confidence(text):
    confidence = {}
    for line in text.split("\n"):
        match = re.match(r"(Bending|Torsional|Tensile|Reversed Bending|Rotating Bending)[:\s]+(\d+)%", line.strip(), re.I)
        if match:
            label = match.group(1).title()
            percent = int(match.group(2))
            confidence[label] = percent
    return confidence

def generate_pdf(case_folder, summary, table):
    pdf_path = os.path.join(case_folder, "report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "ANDALAN FRACTOGRAPHY REPORT")
    c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 710, "Summary:")
    c.setFont("Helvetica", 10)
    text = c.beginText(50, 695)
    for line in summary.split("\n"):
        text.textLine(line)
    c.drawText(text)

    text = c.begin
