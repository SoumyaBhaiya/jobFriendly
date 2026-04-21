from flask import Flask, request, render_template
import pdfplumber

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

def flatten_skills(skill_dict):
    all_skills = []
    for skills in skill_dict.values():
        all_skills.extend(skills)
    return list(set(all_skills))

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


#Adding a Skill Classification
def classify_skills_llm(skill_list):
    if not skill_list:
        return{"must_have": [], "good_to_have": []}
    prompt = f"""
    Classify these skills into:
    - must_have
    - good_to_have

    Return STRICT JSON:

    {{
    "must_have": [],
    "good_to_have": []
    }}

    Skills:
    {skill_list}
    """
    try:
        response = query_llama(prompt)
        return json.loads(response)
    except:
        return {"must_have": skill_list, "good_to_have": []}
    

#an experience socre (for now)

def experience_score_llm(resume_text, job_desc):
    prompt = f"""
    Give a score from  0 to 1 for how well this resume shows relevant experience.

    Resume:
    {resume_text[:1500]}

    Job Description:
    {job_desc}

    ONLY RETURN A NUMBER BETWEEN 0 AND 1.
    Example Output: 0.58 (JUST THE NUMBER)
    """
    try:
        response = query_llama(prompt)
        return float(response.strip())
    except:
        return 0.5
    
model = SentenceTransformer('all-MiniLM-L6-v2')


#UPDATE, NEW COMPUTE SIMILARITY (if works better old one gonna be completely removed)

def compute_similarity(resume_text, job_desc):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_desc, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)
    return float(score)

#adding a 'keyword match score': Quite simple, total skills matched divided by total job skills.
#what extra to add here? 
'''BIGGEST QUESTION IS: WHAT CLASSIFIES AS A 'SKILL'?'''

#Removed the keyword score
#Adding the Weighted Skill Scoring System

def weighted_skill_score(resume_text, must_skills, good_skills):
    resume_text = resume_text.lower()

    must_match = sum (1 for s in must_skills if s.lower() in resume_text)
    good_match = sum(1 for s in good_skills if s.lower() in resume_text)

    must_total = len(must_skills)
    good_total = len(good_skills)

    must_ratio = must_match / must_total if must_total else 0
    good_ratio = good_match / good_total if good_total else 0

    score = (0.7* must_ratio) + (0.3 * good_ratio)

    if must_total > 0 and must_match < (0.5 * must_total):
        score *= 0.6
    
    return score 


#Final Score (more into this in a bit)
def compute_final_score(sem, skill, exp):
    return round((0.5 * sem + 0.3 * skill + 0.2 * exp) * 100, 2)

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
    skill_score = None 
    exp_score = None 
    

    missing_skills = {}
    matched_skills = {} 
    suggestions = [] 

    if request.method == "POST":
        resume = request.files['resume']
        job_desc = request.form["job_desc"].lower()

        resume_text = clean_text(extract_text_from_pdf(resume))
        job_desc_clean = clean_text(job_desc)

        #ADDING A VALIDATION HERE

        if len(job_desc.split()) < 30 or len(resume_text.split()) < 50:
            return render_template("index.html", error = "Provide valid resume and job description.")
        

        #scores

        sem_score = compute_similarity(resume_text, job_desc_clean)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc_clean)

        job_skill_list = flatten_skills(job_skills)
        resume_skill_list = flatten_skills(resume_skills)

        #LLM Classification

        classified = classify_skills_llm(job_skill_list)
        must_skills = classified.get("must_have", [])
        good_skills = classified.get("good_to_have", [])

        #Skill Scoring
        skill_score = weighted_skill_score(resume_text,must_skills, good_skills)

        #experience score
        exp_score = experience_score_llm(resume_text, job_desc)

        #final Score
        
        score = compute_final_score(sem_score, skill_score, exp_score)


        #Matched and missing 
        missing_must = [s for s in must_skills if s.lower() not in resume_text]
        matched_must = [s for s in must_skills if s.lower() in resume_text]

        missing_skills = {"Must Have": missing_must}
        matched_skills = {"Must Have": matched_must}

        suggestions = generate_suggestions(resume_text, job_desc, missing_must)
    
    #returning to webpage 
    
    return render_template('index.html', score=score, missing_skills=missing_skills, 
                           matched_skills=matched_skills, key_score=skill_score, sem_score=sem_score,
                           suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)


