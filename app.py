from flask import Flask, request, render_template
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util  
import requests 
# from ollama import chat ##I don't think i need this now. cause i will use openai localhost server 
import json
import re

app = Flask(__name__)

def extract_text_from_pdf(file):
    text =  ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text.lower()
    

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text) #removing the extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text.strip()

#TO DO:
#AIM: To make a dataset that's gonna update itself with skills in their respective 'domain' as they read more JD and resume
#The Common Skills list is going to be replaced by a dataset or a dict taken from that dataset.
def load_skills():
    with open("skills.json", "r") as f:
        return json.load(f)
COMMON_SKILLS = load_skills()

#UPDATED this function to handle category too
def extract_skills(text):
    found_skills = {category:  [] for category in COMMON_SKILLS}
    for category, skills in COMMON_SKILLS.items():
        for skill in skills:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text):
                found_skills[category].append(skill)
    found_skills = {k: v for k, v in found_skills.items() if v}
    return found_skills

#AI: Model type and connection
def query_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt":prompt,
            "stream": False
        }
    )
    return response.json()["response"]


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
    return float(score)

#adding a 'keyword match score': Quite simple, total skills matched divided by total job skills.
#what extra to add here? 
'''BIGGEST QUESTION IS: WHAT CLASSIFIES AS A 'SKILL'?'''

#UPDATED THIS FUNCTION TO HANDLE CATEOGRY TOO
def keyword_score(resume_skills, job_skills):
    if not job_skills:
        return 0
    match = 0
    total = 0
    for category in job_skills:
        job_set = set(job_skills[category])
        total += len(job_set)
        resume_set = set(resume_skills.get(category, []))
        match+= len(job_set & resume_set)
    if total ==0:
        return 0
    return match / total


#Final Score (more into this in a bit)
def compute_final_score(sem, key):
    return round((0.7 * sem + 0.3 * key) * 100, 2)

#SUGGESTIONS (WILL BE ADDED MORE INTO THIS)
#Adding AI here.
def generate_suggestions(resume_text, job_desc, missing_skills):
    prompt = f"""
    You are a professional resume coach.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}

    Missing Skills:
    {missing_skills}


    Give:
    1. 3 most important improvements
    2. What recruiteer might reject this candidate for

    YOU ARE TO GIVE ONLY ONE LINE ANSWERS. DO NOT GIVE ANYTHING UNNCESSARY
    AND ONLY ANSWER WHAT'S ASKED, NO EXTRA SYLLABLE. (YOU ABSOLUTELY NEED TO GIVE THOSE 2 THINGS)

    An Example Response:
    1) <1: (10-12 words MAX)>
    2) <2: (10-12 words MAX)>
    3) <3: (10-12 words MAX)>

    Recruiter might reject you because:
    <reason here (not more than 20 words) (if there are multiple reason, make them short and seperate them with ',')>
    """
    return query_llama(prompt=prompt)
# WHAT TO CHANGE IN THE PROMPT AREA. Give:
#    1. 3 most important improvements
#    2. What recruiteer might reject this candidate for
# YOU ARE TO GIVE ONLY ONE LINE ANSWERS. DO NOT GIVE ANYTHING UNNCESSARY
# AND ONLY ANSWER WHAT'S ASKED, NO EXTRA SYLLABLE (fine tune this?)

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None 
    sem_score = None 
    key_score = None 
    

    missing_skills = {}
    matched_skills = {} 
    suggestions = [] 

    if request.method == "POST":
        resume = request.files['resume']
        job_desc = request.form["job_desc"].lower()

        resume_text = clean_text(extract_text_from_pdf(resume))
        job_desc_clean = clean_text(job_desc)

        #scores

        sem_score = compute_similarity(resume_text, job_desc_clean)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc_clean)
        key_score = keyword_score(resume_skills, job_skills)

        score = compute_final_score(sem_score, key_score)


        #EDITING THE PART BELOW TO HANDLE CATEGORIES
        #skills
        for category in job_skills:
            matched = set(job_skills[category]) & set(resume_skills.get(category, []))
            missing = set(job_skills[category]) - set(resume_skills.get(category, []))
            if matched:
                matched_skills[category] = list(matched)
            if missing:
                missing_skills[category] = list(missing)
        #suggestions 
        suggestions = generate_suggestions(resume_text, job_desc, missing_skills)
    
    #returning to webpage 
    
    return render_template('index.html', score=score, missing_skills=missing_skills, 
                           matched_skills=matched_skills, key_score=key_score, sem_score=sem_score,
                           suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)


