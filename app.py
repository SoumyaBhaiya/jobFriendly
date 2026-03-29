from flask import Flask, request, render_template
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util 

import re

app = Flask(__name__)

def extract_text_from_pdf(file):
    text =  ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text.lower()
    

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

COMMON_SKILLS = ["excel", "sql", "python", "tableau", "powerbi", "machine learning",
    "data analysis", "statistics", "etl", "dashboard", "reporting",
    "crm", "netsuite", "inventory", "sales", "analytics"]

def extract_skills(text):
    found = []
    for skill in COMMON_SKILLS:
        if skill in text:
            found.append(skill)
    return found 

model = SentenceTransformer('all-MiniLM-L6-v2')

# def compute_similarity(resume_text, job_desc):
#     docs = [resume_text, job_desc]
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(docs)
#     score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
#     return round(score[0][0] * 100, 2)

#UPDATE, NEW COMPUTE SIMILARITY (if works better old one gonna be completely removed)

def compute_similarity(resume_text, job_desc):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_desc, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)
    return round(float(score) * 100, 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    score = None 

    missing_skills = []
    matched_skills = [] 

    if request.method == "POST":
        resume = request.files['resume']
        job_desc = request.form["job_desc"].lower()

        resume_text = clean_text(extract_text_from_pdf(resume))
        job_desc_clean = clean_text(job_desc)

        score = compute_similarity(resume_text, job_desc_clean)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc_clean)

        matched_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))



        
    return render_template('index.html', score=score, missing_skills=missing_skills, matched_skills=matched_skills)


if __name__ == "__main__":
    app.run(debug=True)


