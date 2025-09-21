# dashboard.py - Fixed Integration for Resume Relevance Check System
# Innomatics Research Labs

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import hashlib
import re
from collections import Counter

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Relevance Dashboard | Innomatics Research Labs",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS (same as before but simplified)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .status-high {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .status-low {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .skill-tag {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #90caf9;
    }
    
    .missing-skill-tag {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        color: #c62828;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
    
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ResumeRelevanceDashboard:
    """Main dashboard class for Resume Relevance Check System"""
    
    def __init__(self):
        self.db_path = "resume_system.db"
        self.init_database()
        
        # Initialize session state
        if 'current_job_id' not in st.session_state:
            st.session_state.current_job_id = None
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = []
        if 'jobs_data' not in st.session_state:
            st.session_state.jobs_data = []
    
    def init_database(self):
        """Initialize database tables"""
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Job descriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    job_id TEXT PRIMARY KEY,
                    company TEXT,
                    role_title TEXT,
                    required_skills TEXT,
                    preferred_skills TEXT,
                    experience_required TEXT,
                    education TEXT,
                    location TEXT,
                    description_text TEXT,
                    created_at TIMESTAMP
                )
            """)
            
            # Resumes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    resume_id TEXT PRIMARY KEY,
                    candidate_name TEXT,
                    email TEXT,
                    phone TEXT,
                    skills TEXT,
                    experience TEXT,
                    education TEXT,
                    projects TEXT,
                    certifications TEXT,
                    resume_text TEXT,
                    file_path TEXT,
                    uploaded_at TIMESTAMP
                )
            """)
            
            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    resume_id TEXT,
                    job_id TEXT,
                    relevance_score REAL,
                    hard_match_score REAL,
                    soft_match_score REAL,
                    missing_skills TEXT,
                    matching_skills TEXT,
                    verdict TEXT,
                    suggestions TEXT,
                    evaluated_at TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Basic counts
            cursor.execute("SELECT COUNT(DISTINCT job_id) FROM job_descriptions")
            total_jobs = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(DISTINCT resume_id) FROM resumes")
            total_candidates = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(relevance_score) FROM evaluations")
            avg_score_result = cursor.fetchone()[0]
            avg_score = round(avg_score_result, 2) if avg_score_result else 0
            
            # Score distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN relevance_score >= 75 THEN 'High'
                        WHEN relevance_score >= 50 THEN 'Medium'
                        ELSE 'Low'
                    END as category,
                    COUNT(*) as count
                FROM evaluations
                GROUP BY category
            """)
            score_distribution = dict(cursor.fetchall())
            
            return {
                'total_jobs': total_jobs,
                'total_candidates': total_candidates,
                'total_evaluations': total_evaluations,
                'avg_score': avg_score,
                'score_distribution': score_distribution
            }
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return {
                'total_jobs': 0,
                'total_candidates': 0,
                'total_evaluations': 0,
                'avg_score': 0,
                'score_distribution': {}
            }
        finally:
            conn.close()
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from job description or resume text"""
        common_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'sql', 'nosql',
            'mongodb', 'postgresql', 'mysql', 'react', 'angular', 'vue', 'nodejs',
            'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws', 'azure',
            'gcp', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'machine learning', 'deep learning', 'data science', 'artificial intelligence',
            'git', 'linux', 'agile', 'scrum', 'rest api', 'microservices', 'devops'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def process_job_description(self, jd_text: str, company: str, role_title: str, location: str) -> str:
        """Process and save job description"""
        try:
            job_id = hashlib.md5(jd_text.encode()).hexdigest()[:10]
            
            # Extract skills
            skills = self.extract_skills_from_text(jd_text)
            required_skills = skills[:6]  # First 6 as required
            preferred_skills = skills[6:12]  # Next 6 as preferred
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO job_descriptions 
                (job_id, company, role_title, required_skills, preferred_skills,
                 experience_required, education, location, description_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, company or "Unknown Company", role_title or "Software Engineer",
                json.dumps(required_skills), json.dumps(preferred_skills),
                "2-4 years", json.dumps([]), location or "Remote",
                jd_text, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            return job_id
            
        except Exception as e:
            st.error(f"Error processing job description: {str(e)}")
            return None
    
    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """Get job details from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM job_descriptions WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = ['job_id', 'company', 'role_title', 'required_skills', 'preferred_skills',
                          'experience_required', 'education', 'location', 'description_text', 'created_at']
                result = dict(zip(columns, row))
                result['required_skills'] = json.loads(result['required_skills'])
                result['preferred_skills'] = json.loads(result['preferred_skills'])
                return result
            return None
            
        except Exception as e:
            st.error(f"Error fetching job details: {str(e)}")
            return None
    
    def extract_resume_data(self, file_path: str, filename: str) -> Optional[Dict]:
        """Extract basic data from resume file"""
        try:
            text = ""
            
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text = self.extract_pdf_text(file_path)
            elif filename.lower().endswith('.docx'):
                text = self.extract_docx_text(file_path)
            
            if not text or text.startswith("Error"):
                return None
            
            # Extract basic information
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            phone_pattern = re.compile(r'[\+\(]?[0-9][0-9 .\-\(\)]{8,}[0-9]')
            
            email_match = email_pattern.search(text)
            phone_match = phone_pattern.search(text)
            
            # Extract skills
            skills = self.extract_skills_from_text(text)
            
            return {
                'candidate_name': self.extract_candidate_name(text),
                'email': email_match.group(0) if email_match else '',
                'phone': phone_match.group(0) if phone_match else '',
                'skills': skills,
                'resume_text': text,
                'filename': filename
            }
            
        except Exception as e:
            st.warning(f"Error processing {filename}: {str(e)}")
            return None
    
    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text.strip()
        except ImportError:
            return "Error: PyPDF2 not installed. Run: pip install PyPDF2"
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except ImportError:
            return "Error: python-docx not installed. Run: pip install python-docx"
        except Exception as e:
            return f"Error extracting DOCX: {str(e)}"
    
    def extract_candidate_name(self, text: str) -> str:
        """Extract candidate name from resume"""
        lines = text.split('\n')[:8]
        for line in lines:
            line = line.strip()
            if (5 < len(line) < 50 and 
                not '@' in line and 
                not any(char.isdigit() for char in line) and
                not line.lower().startswith(('resume', 'cv', 'curriculum', 'profile'))):
                return line
        return "Unknown Candidate"
    
    def calculate_relevance_score(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Calculate relevance score"""
        try:
            score = 0
            
            # Get skills
            resume_skills = set(skill.lower() for skill in resume_data.get('skills', []))
            required_skills = set(skill.lower() for skill in job_data.get('required_skills', []))
            preferred_skills = set(skill.lower() for skill in job_data.get('preferred_skills', []))
            
            all_job_skills = required_skills.union(preferred_skills)
            
            # Skills matching (80% weight)
            if all_job_skills:
                matching_skills = resume_skills.intersection(all_job_skills)
                missing_skills = all_job_skills - resume_skills
                
                skill_match_ratio = len(matching_skills) / len(all_job_skills)
                score += skill_match_ratio * 80
                
                # Bonus for required skills
                if required_skills:
                    required_match = len(resume_skills.intersection(required_skills)) / len(required_skills)
                    score += required_match * 15
            else:
                matching_skills = set()
                missing_skills = set()
            
            # Text similarity (5% weight) - simple keyword overlap
            resume_text = resume_data.get('resume_text', '').lower()
            job_text = job_data.get('description_text', '').lower()
            
            resume_words = set(resume_text.split())
            job_words = set(job_text.split())
            common_words = resume_words.intersection(job_words)
            
            if job_words:
                text_similarity = len(common_words) / len(job_words)
                score += text_similarity * 5
            
            # Cap at 100
            score = min(score, 100)
            
            # Determine verdict
            if score >= 75:
                verdict = "High Suitability"
            elif score >= 50:
                verdict = "Medium Suitability"
            else:
                verdict = "Low Suitability"
            
            # Generate suggestions
            suggestions = []
            missing_list = list(missing_skills)
            if missing_list:
                suggestions.append(f"Consider learning: {', '.join(missing_list[:3])}")
            
            if score < 50:
                suggestions.append("Focus on core technical skills mentioned in job description")
                suggestions.append("Add relevant projects demonstrating practical skills")
            elif score < 75:
                suggestions.append("Good potential - work on filling skill gaps")
                suggestions.append("Consider certifications in missing technical areas")
            else:
                suggestions.append("Strong profile! Minor improvements can enhance competitiveness")
            
            return {
                'relevance_score': round(score, 1),
                'hard_match_score': round(score * 0.9, 1),
                'soft_match_score': round(score * 1.1, 1),
                'matching_skills': list(matching_skills),
                'missing_skills': missing_list,
                'verdict': verdict,
                'suggestions': suggestions[:4]
            }
            
        except Exception as e:
            st.error(f"Error calculating score: {str(e)}")
            return {
                'relevance_score': 0,
                'matching_skills': [],
                'missing_skills': [],
                'verdict': "Error in evaluation",
                'suggestions': ["Manual review required"]
            }
    
    def render_header(self):
        """Render header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Resume Relevance Check System</h1>
            <p>Innomatics Research Labs - AI-Powered Talent Matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics(self, stats: Dict):
        """Render metrics dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_jobs']}</div>
                <div class="metric-label">Job Postings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_candidates']}</div>
                <div class="metric-label">Candidates</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_evaluations']}</div>
                <div class="metric-label">Evaluations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['avg_score']}%</div>
                <div class="metric-label">Avg Score</div>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main dashboard function"""
        self.render_header()
        
        # Get stats
        stats = self.get_database_stats()
        self.render_metrics(stats)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Job Upload",
            "📄 Resume Evaluation", 
            "🏆 Results",
            "📊 Analytics"
        ])
        
        with tab1:
            self.render_job_tab()
        
        with tab2:
            self.render_resume_tab()
        
        with tab3:
            self.render_results_tab()
        
        with tab4:
            self.render_analytics_tab()
    
    def render_job_tab(self):
        """Render job upload tab"""
        st.subheader("📋 Job Description Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            with st.form("job_form"):
                company = st.text_input("Company Name", placeholder="e.g., Innomatics Research Labs")
                role_title = st.text_input("Job Title", placeholder="e.g., Python Developer")
                location = st.text_input("Location", placeholder="e.g., Hyderabad")
                jd_text = st.text_area("Job Description", height=200, 
                                      placeholder="Paste the complete job description here...")
                
                submit_jd = st.form_submit_button("🚀 Process Job Description", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submit_jd and jd_text.strip():
                with st.spinner("Processing..."):
                    job_id = self.process_job_description(jd_text, company, role_title, location)
                    if job_id:
                        st.session_state.current_job_id = job_id
                        st.success("✅ Job processed successfully!")
                        st.rerun()
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Current Status")
            
            if st.session_state.current_job_id:
                job_details = self.get_job_details(st.session_state.current_job_id)
                if job_details:
                    st.markdown('<div class="status-high">✅ Ready for Evaluation</div>', unsafe_allow_html=True)
                    st.write(f"**Role:** {job_details['role_title']}")
                    st.write(f"**Company:** {job_details['company']}")
                    st.write(f"**Location:** {job_details['location']}")
                    
                    if job_details['required_skills']:
                        st.write("**Required Skills:**")
                        skills_html = ""
                        for skill in job_details['required_skills'][:8]:
                            skills_html += f'<span class="skill-tag">{skill}</span> '
                        st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-low">⚠️ No job loaded</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_resume_tab(self):
        """Render resume evaluation tab"""
        st.subheader("📄 Resume Evaluation")
        
        if not st.session_state.current_job_id:
            st.warning("⚠️ Please upload a job description first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload Resume Files",
                type=['pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload PDF or DOCX files"
            )
            
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} files uploaded")
                
                min_score = st.slider("Minimum Score Filter", 0, 100, 50)
                
                if st.button("🎯 Evaluate Resumes", type="primary", use_container_width=True):
                    self.process_resumes(uploaded_files, min_score)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card" style="height: 300px; padding: 1.5rem;">
                <h4 style="color: #2d3748; margin-bottom: 1rem;">📋 Evaluation Guide</h4>
                <div style="line-height: 2; color: #4a5568;">
                    <strong>📁 Upload:</strong> Multiple PDF/DOCX files<br>
                    <strong>🎯 Filter:</strong> Set minimum score threshold<br>
                    <strong>⚡ Process:</strong> Get instant relevance scores<br>
                    <strong>📊 Analyze:</strong> View detailed skill analysis<br>
                    <strong>🏆 Results:</strong> Ranked candidate list
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def process_resumes(self, uploaded_files, min_score):
        """Process and evaluate resumes"""
        job_details = self.get_job_details(st.session_state.current_job_id)
        if not job_details:
            st.error("Job details not found!")
            return
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                # Save file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract resume data
                resume_data = self.extract_resume_data(temp_path, uploaded_file.name)
                
                if resume_data and not resume_data['resume_text'].startswith("Error"):
                    # Calculate score
                    evaluation = self.calculate_relevance_score(resume_data, job_details)
                    
                    if evaluation['relevance_score'] >= min_score:
                        result = {
                            'candidate_name': resume_data['candidate_name'],
                            'email': resume_data['email'],
                            'phone': resume_data['phone'],
                            'relevance_score': evaluation['relevance_score'],
                            'hard_match_score': evaluation['hard_match_score'],
                            'soft_match_score': evaluation['soft_match_score'],
                            'matching_skills': evaluation['matching_skills'],
                            'missing_skills': evaluation['missing_skills'],
                            'verdict': evaluation['verdict'],
                            'suggestions': evaluation['suggestions']
                        }
                        results.append(result)
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            # Sort by score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Update session state
            st.session_state.evaluation_results = results
            
            # Show completion
            status_text.success(f"✅ Complete! {len(results)} candidates above {min_score}% threshold")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def render_results_tab(self):
        """Render results tab"""
        st.subheader("🏆 Evaluation Results")
        
        if not st.session_state.evaluation_results:
            st.info("📋 No results yet. Upload and evaluate resumes first.")
            return
        
        results = st.session_state.evaluation_results
        df = pd.DataFrame(results)
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        
        high_count = len(df[df['relevance_score'] >= 75])
        medium_count = len(df[(df['relevance_score'] >= 50) & (df['relevance_score'] < 75)])
        low_count = len(df[df['relevance_score'] < 50])
        
        with col1:
            st.markdown(f'<div class="status-high">{high_count}<br>High Suitability</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="status-medium">{medium_count}<br>Medium Suitability</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="status-low">{low_count}<br>Low Suitability</div>', unsafe_allow_html=True)
        with col4:
            avg_score = df['relevance_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        # Detailed results
        st.subheader("📊 Candidate Details")
        
        for _, candidate in df.iterrows():
            score = candidate['relevance_score']
            
            if score >= 75:
                icon = "🌟"
            elif score >= 50:
                icon = "⚡"
            else:
                icon = "⚠️"
            
            with st.expander(f"{icon} {candidate['candidate_name']} - {score}%"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📧 Email:** {candidate['email']}")
                    st.write(f"**📱 Phone:** {candidate['phone']}")
                    st.write(f"**🎯 Verdict:** {candidate['verdict']}")
                    
                    st.write("**✅ Matching Skills:**")
                    if candidate['matching_skills']:
                        skills_html = ""
                        for skill in candidate['matching_skills'][:10]:
                            skills_html += f'<span class="skill-tag">{skill}</span> '
                        st.markdown(skills_html, unsafe_allow_html=True)
                    
                    st.write("**❌ Missing Skills:**")
                    if candidate['missing_skills']:
                        missing_html = ""
                        for skill in candidate['missing_skills'][:10]:
                            missing_html += f'<span class="missing-skill-tag">{skill}</span> '
                        st.markdown(missing_html, unsafe_allow_html=True)
                
                with col2:
                    if score >= 75:
                        score_class = "status-high"
                    elif score >= 50:
                        score_class = "status-medium"
                    else:
                        score_class = "status-low"
                    
                    st.markdown(f'<div class="{score_class}">{score}%<br>Relevance Score</div>', unsafe_allow_html=True)
                    
                    if 'hard_match_score' in candidate:
                        st.write(f"**Hard Match:** {candidate['hard_match_score']}%")
                        st.progress(candidate['hard_match_score'] / 100)
                    
                    if 'soft_match_score' in candidate:
                        st.write(f"**Semantic Match:** {candidate['soft_match_score']}%")
                        st.progress(candidate['soft_match_score'] / 100)
                
                # Suggestions
                if candidate['suggestions']:
                    st.write("**💡 Suggestions:**")
                    for i, suggestion in enumerate(candidate['suggestions'], 1):
                        st.write(f"{i}. {suggestion}")
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.subheader("📊 Analytics & Insights")
        
        if not st.session_state.evaluation_results:
            st.info("📈 Analytics will appear after evaluating resumes.")
            return
        
        df = pd.DataFrame(st.session_state.evaluation_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution pie chart
            df['score_range'] = pd.cut(
                df['relevance_score'],
                bins=[0, 25, 50, 75, 100],
                labels=['0-25%', '26-50%', '51-75%', '76-100%']
            )
            
            score_counts = df['score_range'].value_counts().reset_index()
            score_counts.columns = ['Score Range', 'Count']
            
            fig = px.pie(
                score_counts,
                values='Count',
                names='Score Range',
                title="Score Distribution",
                color_discrete_sequence=['#f56565', '#ed8936', '#48bb78', '#38a169']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Skills gap analysis
            all_missing_skills = []
            for result in st.session_state.evaluation_results:
                all_missing_skills.extend(result.get('missing_skills', []))
            
            if all_missing_skills:
                skill_counts = Counter(all_missing_skills)
                top_missing = skill_counts.most_common(10)
                
                missing_df = pd.DataFrame(top_missing, columns=['Skill', 'Frequency'])
                
                fig = px.bar(
                    missing_df,
                    x='Frequency',
                    y='Skill',
                    orientation='h',
                    title="Top Missing Skills",
                    color='Frequency',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        st.subheader("📈 Detailed Statistics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**🎯 Top Matching Skills**")
            all_matching_skills = []
            for result in st.session_state.evaluation_results:
                all_matching_skills.extend(result.get('matching_skills', []))
            
            if all_matching_skills:
                matching_counts = Counter(all_matching_skills)
                top_matching = matching_counts.most_common(8)
                
                for skill, count in top_matching:
                    percentage = (count / len(st.session_state.evaluation_results)) * 100
                    st.write(f"**{skill}:** {count} candidates ({percentage:.1f}%)")
        
        with col4:
            st.write("**📊 Score Statistics**")
            scores = df['relevance_score']
            
            st.metric("Highest Score", f"{scores.max():.1f}%")
            st.metric("Lowest Score", f"{scores.min():.1f}%")
            st.metric("Median Score", f"{scores.median():.1f}%")
            st.metric("Standard Deviation", f"{scores.std():.1f}")


# Main function that can be imported and called
def run_streamlit_app():
    """Main function to run the Streamlit dashboard"""
    try:
        dashboard = ResumeRelevanceDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        st.info("Please ensure all required packages are installed: pip install streamlit plotly pandas PyPDF2 python-docx")


# For direct execution
if __name__ == "__main__":
    run_streamlit_app()