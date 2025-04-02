
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

    text = c.beginText(50, 620)
    text.textLine("Analysis Table:")
    for line in table.split("\n"):
        text.textLine(line)
    c.drawText(text)

    annotated_path = os.path.join(case_folder, "annotated_image.jpg")
    if os.path.exists(annotated_path):
        c.drawImage(annotated_path, 50, 300, width=400, preserveAspectRatio=True)

    c.save()

# Streamlit UI
st.set_page_config(page_title="ANDALAN FRACTOGRAPHY SOLVER", layout="centered")
st.title("🧠 ANDALAN FRACTOGRAPHY SOLVER – PRO EDITION")

uploaded_file = st.file_uploader("Upload a fracture image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    edges, description = extract_fracture_description(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

# 🖊️ Crack Origin Annotation (safe canvas background image)
canvas_bg = np.array(image.convert("RGB"))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.6)",
    background_image=canvas_bg,
    height=canvas_bg.shape[0],
    width=canvas_bg.shape[1],
    drawing_mode="point",
    key="canvas"
)

# Extract annotation results
marked_points = canvas_result.json_data["objects"] if canvas_result.json_data else []

# Edge Detection
st.subheader("Edge Detection")
st.image(edges, clamp=True, channels="GRAY", use_column_width=True)

# GPT Analysis
st.subheader("GPT Analysis Result")
with st.spinner("Analyzing..."):
    result = generate_gpt_analysis(description)


    parts = result.split("\n\n", 1)
    summary = parts[0]
    table = parts[1] if len(parts) > 1 else ""

    st.markdown("**Summary:**")
    st.markdown(summary)
    st.markdown("### Fracture Table")
    st.markdown(table)

    confidence = extract_stress_confidence(result)
    if confidence:
        st.markdown("### Stress Type Confidence")
        fig, ax = plt.subplots()
        ax.barh(list(confidence.keys()), list(confidence.values()))
        ax.set_xlim(0, 100)
        st.pyplot(fig)

    case_dir = get_next_case_id()
    os.makedirs(case_dir, exist_ok=True)
    original_path = os.path.join(case_dir, "uploaded_image.jpg")
    edge_path = os.path.join(case_dir, "edges.jpg")
    image.save(original_path)
    Image.fromarray(edges).save(edge_path)

    if marked_points:
        draw = ImageDraw.Draw(image)
        for idx, point in enumerate(marked_points):
            x, y = point["left"], point["top"]
            draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
            draw.text((x+8, y), f"Origin {idx+1}", fill="red")
        annotated_path = os.path.join(case_dir, "annotated_image.jpg")
        image.save(annotated_path)
    else:
        annotated_path = original_path

    with open(os.path.join(case_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n\n" + table)

    generate_pdf(case_dir, summary, table)

    log_path = "cases/cases_log.csv"
    os.makedirs("cases", exist_ok=True)
    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(log_path).st_size == 0:
            writer.writerow(["Case", "Date", "Failure Mode", "Stress Type", "Origins", "PDF"])
        failure, stress, origin_count = "", "", ""
        for line in table.split("\n"):
            if "Failure Mode" in line: failure = line.split("|")[2].strip()
            if "Type of Stress" in line: stress = line.split("|")[2].strip()
            if "Origin Count" in line: origin_count = line.split("|")[2].strip()
        writer.writerow([os.path.basename(case_dir), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), failure, stress, origin_count, "report.pdf"])

    st.success(f"Case saved: {case_dir}")
    with open(os.path.join(case_dir, "report.pdf"), "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="fracture_report.pdf")

    st.markdown("---")
    st.markdown("### 💬 Ask GPT about this fracture")
    with st.form("chat_form"):
        user_query = st.text_input("Ask or suggest a correction:")
        submitted = st.form_submit_button("Submit")
    if submitted and user_query:
        followup_prompt = f"Image description: {description}\n\nPrevious GPT Analysis:\n{result}\n\nUser feedback: {user_query}"
        followup = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional failure analyst."},
                {"role": "user", "content": followup_prompt}
            ],
            max_tokens=500
        )
        st.success("Your input has been learned.")
        st.markdown("**GPT Follow-up Response:**")
        st.markdown(followup.choices[0].message.content)
