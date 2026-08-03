from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import sqlite3
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder=os.path.abspath('templates'))
app.secret_key = 'super_secret_placement_key'
base_dir = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(base_dir, 'database', 'placement.db')

# Load the model
MODEL_PATH = os.path.join('model', 'model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: Model not found. Run train_model.py first.")

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs('database', exist_ok=True)
    with get_db() as conn:
        cursor = conn.cursor()
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        # Recreate predictions table to perfectly match the 15 features
        cursor.execute('DROP TABLE IF EXISTS predictions')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                email TEXT,
                branch TEXT,
                cgpa REAL,
                backlogs INTEGER DEFAULT 0,
                coding_skills INTEGER DEFAULT 0,
                dsa_score INTEGER DEFAULT 0,
                aptitude_score INTEGER DEFAULT 0,
                communication_skills INTEGER DEFAULT 0,
                ml_knowledge INTEGER DEFAULT 0,
                system_design INTEGER DEFAULT 0,
                internships INTEGER DEFAULT 0,
                projects_count INTEGER DEFAULT 0,
                certifications INTEGER DEFAULT 0,
                hackathons INTEGER DEFAULT 0,
                open_source_contributions INTEGER DEFAULT 0,
                extracurriculars INTEGER DEFAULT 0,
                prediction_status TEXT,
                probability REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()

# Initialize DB on startup
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']  # In production, hash the password!
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, password))
                conn.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
            user = cursor.fetchone()
            
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('Session expired. Please login again.', 'warning')
        return redirect(url_for('login'))
    
    if not model:
        flash('Machine learning model is not available.', 'danger')
        return redirect(url_for('dashboard'))

    # Extract exactly the required fields
    name = request.form.get('name')
    email = request.form.get('email')
    
    branch = request.form.get('branch', 'Unknown')
    cgpa = float(request.form.get('cgpa', 0))
    backlogs = int(request.form.get('backlogs', 0))
    coding_skills = int(request.form.get('coding_skills', 0))
    dsa_score = int(request.form.get('dsa_score', 0))
    aptitude_score = int(request.form.get('aptitude_score', 0))
    communication_skills = int(request.form.get('communication_skills', 0))
    ml_knowledge = int(request.form.get('ml_knowledge', 0))
    system_design = int(request.form.get('system_design', 0))
    internships = int(request.form.get('internships', 0))
    projects_count = int(request.form.get('projects_count', 0))
    certifications = int(request.form.get('certifications', 0))
    hackathons = int(request.form.get('hackathons', 0))
    open_source_contributions = int(request.form.get('open_source_contributions', 0))
    extracurriculars = int(request.form.get('extracurriculars', 0))
    
    # Create a DataFrame for robust prediction
    feature_dict = {
        'branch': [branch],
        'cgpa': [cgpa],
        'backlogs': [backlogs],
        'coding_skills': [coding_skills],
        'dsa_score': [dsa_score],
        'aptitude_score': [aptitude_score],
        'communication_skills': [communication_skills],
        'ml_knowledge': [ml_knowledge],
        'system_design': [system_design],
        'internships': [internships],
        'projects_count': [projects_count],
        'certifications': [certifications],
        'hackathons': [hackathons],
        'open_source_contributions': [open_source_contributions],
        'extracurriculars': [extracurriculars]
    }
    
    df_pred = pd.DataFrame(feature_dict)
    
    prediction_class = model.predict(df_pred)[0]
    probabilities = model.predict_proba(df_pred)[0]
    
    # Get probability of positive class (Placed)
    prob_score = round(probabilities[1] * 100, 2)
    
    # Penalty for backlogs
    if backlogs > 0:
        prob_score = max(0, prob_score - (backlogs * 15))

    if prob_score >= 75:
        status = "Placed"
    elif prob_score >= 40:
        status = "Likely to be Placed"
    else:
        status = "Needs Improvement"
    
    # Save to database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (user_id, name, email, branch, cgpa, backlogs, coding_skills, dsa_score, aptitude_score, communication_skills, ml_knowledge, system_design, internships, projects_count, certifications, hackathons, open_source_contributions, extracurriculars, prediction_status, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], name, email, branch, cgpa, backlogs, coding_skills, dsa_score, aptitude_score, communication_skills, ml_knowledge, system_design, internships, projects_count, certifications, hackathons, open_source_contributions, extracurriculars, status, prob_score))
        conn.commit()
    
    # Calculate domain scores using the specific inputs
    domain_scores = {
        'Programming': f"{int((coding_skills/100) * 7)}/7",
        'AI / ML': f"{int((ml_knowledge/100) * 4)}/4",
        'System Design': f"{int((system_design/100) * 5)}/5",
        'Aptitude': f"{int((aptitude_score/100) * 3)}/3"
    }

    # Pass data to result page
    result_data = {
        'name': name,
        'status': status,
        'probability': prob_score,
        'cgpa': cgpa,
        'backlogs': backlogs,
        'coding_skills': coding_skills,
        'dsa_score': dsa_score,
        'internships': internships,
        'projects_count': projects_count,
        'communication_skills': communication_skills,
        'aptitude_score': aptitude_score,
        'ml_knowledge': ml_knowledge,
        'system_design': system_design,
        'certifications': certifications,
        'hackathons': hackathons,
        'open_source_contributions': open_source_contributions,
        'extracurriculars': extracurriculars,
        'missing_skills': [],
        'domain_scores': domain_scores
    }

    # Company Recommendations Logic
    company_recommendations = []
    
    if prob_score >= 80 and dsa_score >= 80 and system_design >= 60:
        company_recommendations.extend([
            {"name": "Google", "logo": "bi-google", "color": "#ea4335", "focus": "Advanced DSA & System Design", "link": "https://careers.google.com/"},
            {"name": "Microsoft", "logo": "bi-windows", "color": "#00a4ef", "focus": "DSA, OOP & System Design", "link": "https://careers.microsoft.com/"},
            {"name": "Amazon", "logo": "bi-box-seam", "color": "#ff9900", "focus": "DSA, Leadership Principles & AWS", "link": "https://amazon.jobs/"}
        ])
    
    elif prob_score >= 65 and coding_skills >= 70:
        company_recommendations.extend([
            {"name": "Adobe", "logo": "bi-palette", "color": "#ff0000", "focus": "Strong CS Fundamentals & Coding", "link": "https://careers.adobe.com/"},
            {"name": "Atlassian", "logo": "bi-layers", "color": "#0052cc", "focus": "DSA, Web Tech & Problem Solving", "link": "https://www.atlassian.com/company/careers"},
            {"name": "Uber", "logo": "bi-car-front", "color": "#ffffff", "focus": "System Design, Microservices, DSA", "link": "https://www.uber.com/us/en/careers/"}
        ])

    elif prob_score >= 40:
        company_recommendations.extend([
            {"name": "TCS", "logo": "bi-building", "color": "#00a1e0", "focus": "Aptitude, Core Java/Python, Database", "link": "https://www.tcs.com/careers"},
            {"name": "Infosys", "logo": "bi-buildings", "color": "#007cc3", "focus": "Aptitude, Basic Programming, DBMS", "link": "https://www.infosys.com/careers"},
            {"name": "Accenture", "logo": "bi-globe", "color": "#a100ff", "focus": "Communication, Aptitude, Basic Coding", "link": "https://www.accenture.com/us-en/careers"}
        ])

    if ml_knowledge >= 70:
         company_recommendations.append({"name": "OpenAI / AI Labs", "logo": "bi-robot", "color": "#10a37f", "focus": "Deep Learning, Python, NLP", "link": "https://openai.com/careers/"})
         
    if len(company_recommendations) == 0:
        company_recommendations.extend([
            {"name": "Wipro", "logo": "bi-building", "color": "#ffffff", "focus": "Aptitude, Basic Communication", "link": "https://careers.wipro.com/"},
            {"name": "Cognizant", "logo": "bi-building", "color": "#0033a0", "focus": "Aptitude, Basic Problem Solving", "link": "https://careers.cognizant.com/global/en"}
        ])
        
    seen_companies = set()
    unique_companies = []
    for comp in company_recommendations:
        if comp['name'] not in seen_companies:
            unique_companies.append(comp)
            seen_companies.add(comp['name'])
            
    result_data['company_recommendations'] = unique_companies[:4]
    
    return render_template('result.html', data=result_data)

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email == 'admin@admin.com' and password == 'admin123':
            session['is_admin'] = True
            session['user_name'] = 'Administrator'
            flash('Admin access granted.', 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid Admin Credentials.', 'danger')
            
    return render_template('admin_login.html')

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        flash('Unauthorized access. Please login as admin.', 'danger')
        return redirect(url_for('admin_login'))
        
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions ORDER BY id DESC')
        predictions = cursor.fetchall()
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) as total_students, AVG(cgpa) as avg_cgpa FROM predictions')
        stats = cursor.fetchone()

        cursor.execute('SELECT id, name, email FROM users ORDER BY id DESC')
        registered_users = cursor.fetchall()
        
    return render_template('admin.html', predictions=predictions, stats=stats, users=registered_users)

@app.route('/download_report')
def download_report():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
        
    with get_db() as conn:
        df = pd.read_sql_query("SELECT * from predictions", conn)
    
    report_path = 'database/placement_report.csv'
    df.to_csv(report_path, index=False)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    # Grab port assigned by hosting provider, default to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode for production deployment
    app.run(host='0.0.0.0', port=port, debug=False)

# Trigger server restart
