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
    text = text.lower()
    text = re.sub(r"\s+", " ", text) #removing the extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text.strip()

#TO DO:
#AIM: To make a dataset that's gonna update itself with skills in their respective 'domain' as they read more JD and resume
#The Common Skills list is going to be replaced by a dataset or a dict taken from that dataset.
COMMON_SKILLS = {
  "Programming Languages": ["Python", "Java", "C++", "C#", "JavaScript", "R", "Ruby", "Go", "PHP", "Swift", "Kotlin", "TypeScript", "Scala", "Perl", "MATLAB", "SQL", "Dart", "Objective-C", "Rust", "Shell Scripting", "VBA", "HTML", "CSS", "SAS", "Groovy", "Lua"],
  
  "Data Analysis & Visualization": ["Excel", "Tableau", "Power BI", "Pandas", "NumPy", "Matplotlib", "Seaborn", "ggplot2", "Plotly", "QlikView", "Looker", "SPSS", "SAS", "SQL", "BigQuery", "Google Analytics", "RStudio", "Jupyter Notebook", "D3.js", "Alteryx", "Datawrapper", "Excel PivotTables", "Power Query", "Google Data Studio", "Apache Superset", "Zoho Analytics"],
  
  "Machine Learning & AI": ["Scikit-learn", "TensorFlow", "Keras", "PyTorch", "XGBoost", "LightGBM", "CatBoost", "OpenCV", "NLTK", "spaCy", "Hugging Face Transformers", "Fast.ai", "MLlib", "YOLO", "Reinforcement Learning", "Deep Learning", "Computer Vision", "Natural Language Processing", "PCA", "Clustering", "Decision Trees", "Random Forest", "SVM", "Neural Networks", "Gradient Boosting", "AutoML"],
  
  "Cloud & DevOps": ["AWS", "Azure", "Google Cloud Platform", "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD", "Jenkins", "GitHub Actions", "Azure DevOps", "CloudFormation", "Serverless", "Microservices", "Linux", "Linux Bash", "Prometheus", "Grafana", "Nagios", "VMware", "OpenShift", "ELK Stack", "Helm", "ArgoCD", "Cloud Security", "Terraform Modules"],
  
  "Web Development": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Ruby on Rails", "Spring Boot", "Bootstrap", "Tailwind CSS", "SASS", "REST APIs", "GraphQL", "jQuery", "Next.js", "Nuxt.js", "PHP", "ASP.NET", "WordPress", "WebSockets", "Express.js", "MongoDB", "MySQL"],
  
  "Databases & Data Engineering": ["SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Oracle", "Cassandra", "Redis", "BigQuery", "Snowflake", "Redshift", "ETL", "Airflow", "Kafka", "Hadoop", "Spark", "Data Warehousing", "Data Lakes", "DBT", "Elasticsearch", "Presto", "Hive", "MariaDB", "SQLite", "Firebase", "DynamoDB"],
  
  "Project Management & Productivity": ["Agile", "Scrum", "Kanban", "Jira", "Trello", "Asana", "Basecamp", "Monday.com", "ClickUp", "Microsoft Project", "Confluence", "Notion", "Slack", "Teamwork", "Time Management", "Risk Management", "Stakeholder Management", "Task Prioritization", "Gantt Charts", "Resource Allocation", "Project Planning", "Budgeting", "Reporting", "Change Management", "Scope Management"],
  
  "Design & Creativity": ["Adobe Photoshop", "Adobe Illustrator", "Adobe InDesign", "Figma", "Sketch", "Canva", "UI/UX Design", "Wireframing", "Prototyping", "Adobe XD", "InVision", "3D Modeling", "Blender", "After Effects", "Animation", "Typography", "Color Theory", "Design Thinking", "Branding", "Illustration", "Visual Communication", "Motion Graphics", "Photo Editing", "Layout Design", "Interaction Design"],
  
  "Soft Skills": ["Communication", "Leadership", "Teamwork", "Problem-solving", "Critical Thinking", "Adaptability", "Creativity", "Time Management", "Conflict Resolution", "Decision Making", "Emotional Intelligence", "Negotiation", "Presentation Skills", "Networking", "Active Listening", "Collaboration", "Stress Management", "Interpersonal Skills", "Flexibility", "Empathy", "Persuasion", "Motivation", "Mentoring", "Influence", "Public Speaking"],
  
  "Marketing & Sales": ["SEO", "SEM", "Content Marketing", "Social Media Marketing", "Email Marketing", "Google Analytics", "Facebook Ads", "Instagram Ads", "LinkedIn Marketing", "PPC", "Google Ads", "Marketing Strategy", "Brand Management", "CRM", "Salesforce", "Lead Generation", "Copywriting", "Affiliate Marketing", "Conversion Rate Optimization", "Influencer Marketing", "Market Research", "Product Marketing", "Event Marketing", "Campaign Management", "Marketing Automation"],
  
  "Finance & Accounting": ["Financial Analysis", "Budgeting", "Forecasting", "Accounting", "QuickBooks", "SAP", "ERP Systems", "Excel Financial Modeling", "Taxation", "Auditing", "Bookkeeping", "Payroll", "Accounts Payable", "Accounts Receivable", "Financial Reporting", "Cost Analysis", "Cash Flow Management", "Investment Analysis", "Financial Planning", "Variance Analysis", "Risk Management", "Asset Management", "Balance Sheet", "Income Statement", "Ledger Management"],
  
  "Legal & Compliance": ["Contract Management", "Regulatory Compliance", "Risk Assessment", "Corporate Law", "Intellectual Property", "Data Privacy", "GDPR", "HIPAA", "Labor Law", "Employment Law", "Policy Development", "Legal Research", "Case Management", "Litigation Support", "Due Diligence", "Compliance Auditing", "Internal Auditing", "Ethics Management", "Audit Reporting", "Trademark Law", "Patent Law", "Negotiation", "Legal Drafting", "Fraud Detection", "Anti-Money Laundering"],
  
  "Healthcare & Life Sciences": ["Clinical Research", "Patient Care", "Medical Coding", "EMR/EHR", "Healthcare Management", "Pharmacy", "Nursing", "Public Health", "Epidemiology", "Biostatistics", "Laboratory Skills", "Medical Imaging", "HIPAA Compliance", "Disease Management", "Nutrition", "Health Education", "Medical Writing", "Pharmacology", "Diagnostics", "Telemedicine", "Clinical Trials", "Pathology", "Medical Billing", "Surgical Assistance", "Genomics"],
  
  "Languages & Communication": ["English", "Spanish", "French", "German", "Mandarin", "Japanese", "Portuguese", "Russian", "Arabic", "Hindi", "Italian", "Korean", "Sign Language", "Translation", "Interpretation", "Public Speaking", "Copywriting", "Technical Writing", "Editing", "Proofreading", "Presentation", "Storytelling", "Cross-Cultural Communication", "Writing Reports", "Content Creation"],
  
  "Other Technical Skills": ["Linux", "Git", "GitHub", "Command Line", "Networking", "Cybersecurity", "Penetration Testing", "Firewall Management", "VPN Configuration", "Cloud Security", "Encryption", "System Administration", "Troubleshooting", "Virtualization", "Active Directory", "IT Support", "ITIL", "Scripting", "REST APIs", "Automation", "Monitoring Tools", "Backups", "Disaster Recovery", "Performance Tuning", "Patch Management"]
}



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

def generate_suggestions(missing_skills):
    suggestions = []
    for category, skills in missing_skills.items():
        suggestions.append(f"In {category}, Consider adding or learning these skills: {', '.join(skills)}") #made small fix here
    return suggestions 


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
        suggestions = generate_suggestions(missing_skills)
    
    #returning to webpage 
    
    return render_template('index.html', score=score, missing_skills=missing_skills, 
                           matched_skills=matched_skills, key_score=key_score, sem_score=sem_score,
                           suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)


