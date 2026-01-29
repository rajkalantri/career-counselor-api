import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import PyPDF2
import io
import base64
import requests
import hashlib

app = Flask(__name__)
CORS(app)

# Get API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'pcsk_1qJBT_DNs7D7rc6kZRA5rV1jn4CB6R4QjiaJSh1x987DACmgRZkhPpAipUScDCEWwTppv')
JSEARCH_API_KEY = os.environ.get('JSEARCH_API_KEY', '83d8d51b2emsh7752ed56cb2e48ep16f6dcjsndc2626e54409')
JSEARCH_HOST = "jsearch.p.rapidapi.com"

# Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("career-guidance")
print("Pinecone ready!")

# Simple embedding cache to avoid repeated API calls
embedding_cache = {}

def get_embedding(text):
    """Get embedding using Pinecone's inference API"""
    # Check cache first
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Use Pinecone inference API with llama-text-embed-v2 at 384 dimensions
    # (matches our existing Pinecone index which uses 384-dim vectors)
    try:
        response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={
                "input_type": "query",
                "dimension": 384  # Match existing index dimension
            }
        )
        embedding = response.data[0].values
        embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        # Fallback: return None and skip vector search
        return None

# Common tech skills to look for in resumes
KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin", "r", "scala", "matlab",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "opencv", "hugging face", "transformers", "deep learning", "machine learning", "neural networks", "nlp", "computer vision", "reinforcement learning",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery", "data warehousing", "etl",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "github actions", "linux", "bash",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "rest api", "graphql", "html", "css",
    "git", "jira", "confluence", "tableau", "power bi", "excel", "jupyter",
    "communication", "leadership", "teamwork", "problem solving", "agile", "scrum"
]


@app.route('/search', methods=['POST'])
def search():
    """Search Pinecone for relevant career information"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get embedding
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            return jsonify({
                'query': query,
                'results': [],
                'error': 'Could not generate embedding'
            })
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'score': match['score'],
                'type': match['metadata'].get('type', 'unknown'),
                'title': match['metadata'].get('title', match['metadata'].get('name', 'N/A')),
                'text': match['metadata'].get('text', '')
            })
        
        return jsonify({
            'query': query,
            'results': formatted_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/extract-resume', methods=['POST'])
def extract_resume():
    """Extract text and skills from uploaded PDF resume"""
    try:
        data = request.json
        pdf_base64 = data.get('pdf_base64', '')
        
        if not pdf_base64:
            return jsonify({'error': 'No PDF provided'}), 400
        
        # Remove data URL prefix if present
        if ',' in pdf_base64:
            pdf_base64 = pdf_base64.split(',')[1]
        
        # Decode base64 to bytes
        pdf_bytes = base64.b64decode(pdf_base64)
        
        # Extract text from PDF
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Extract skills from resume text
        text_lower = text.lower()
        found_skills = []
        
        for skill in KNOWN_SKILLS:
            if skill.lower() in text_lower:
                found_skills.append(skill.title())
        
        # Remove duplicates and sort
        found_skills = sorted(list(set(found_skills)))
        
        return jsonify({
            'success': True,
            'text': text,
            'extracted_skills': found_skills,
            'skill_count': len(found_skills)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-skills', methods=['POST'])
def analyze_skills():
    """Compare user skills with job requirements and identify gaps"""
    try:
        data = request.json
        user_skills = data.get('user_skills', [])
        target_career = data.get('target_career', 'Machine Learning Engineer')
        
        # Get embedding for career search
        query_embedding = get_embedding(target_career)
        
        if query_embedding is None:
            return jsonify({'error': 'Could not generate embedding'}), 500
        
        # Search for the target career in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Extract required skills from career data
        required_skills = []
        career_info = {}
        
        for match in results['matches']:
            text = match['metadata'].get('text', '')
            
            if 'Technical Skills Required:' in text:
                skills_section = text.split('Technical Skills Required:')[1]
                skills_section = skills_section.split('\n')[0]
                skills = [s.strip() for s in skills_section.split(',')]
                required_skills.extend(skills)
            
            if match['metadata'].get('type') == 'career':
                career_info = {
                    'title': match['metadata'].get('title', target_career),
                    'salary_entry': match['metadata'].get('salary_entry', 'N/A'),
                    'salary_senior': match['metadata'].get('salary_senior', 'N/A')
                }
        
        # Normalize skills for comparison
        user_skills_lower = [s.lower().strip() for s in user_skills]
        
        # Find matches and gaps
        skills_have = []
        skills_need = []
        
        for skill in required_skills:
            if skill.lower().strip() in user_skills_lower:
                skills_have.append(skill)
            else:
                skills_need.append(skill)
        
        skills_have = list(set(skills_have))
        skills_need = list(set(skills_need))
        
        return jsonify({
            'success': True,
            'target_career': career_info.get('title', target_career),
            'skills_you_have': skills_have,
            'skills_you_need': skills_need,
            'match_percentage': round(len(skills_have) / max(len(required_skills), 1) * 100, 1),
            'career_info': career_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search-jobs', methods=['POST'])
def search_jobs():
    """Search for real job listings using JSearch API"""
    try:
        data = request.json
        query = data.get('query', 'software engineer')
        location = data.get('location', '')
        num_results = data.get('num_results', 5)
        
        search_query = query
        if location:
            search_query = f"{query} in {location}"
        
        url = "https://jsearch.p.rapidapi.com/search"
        
        headers = {
            "X-RapidAPI-Key": JSEARCH_API_KEY,
            "X-RapidAPI-Host": JSEARCH_HOST
        }
        
        params = {
            "query": search_query,
            "page": "1",
            "num_pages": "1",
            "country": "us",
            "date_posted": "month"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': f"API error: {response.status_code}",
                'jobs': []
            })
        
        result = response.json()
        
        jobs = []
        for job in result.get('data', [])[:num_results]:
            jobs.append({
                'title': job.get('job_title', 'N/A'),
                'company': job.get('employer_name', 'N/A'),
                'location': job.get('job_city', '') + ', ' + job.get('job_state', '') if job.get('job_city') else job.get('job_country', 'N/A'),
                'salary_min': job.get('job_min_salary'),
                'salary_max': job.get('job_max_salary'),
                'job_type': job.get('job_employment_type', 'N/A'),
                'posted_date': job.get('job_posted_at_datetime_utc', 'N/A')[:10] if job.get('job_posted_at_datetime_utc') else 'N/A',
                'apply_link': job.get('job_apply_link', ''),
                'description_snippet': job.get('job_description', '')[:300] + '...' if job.get('job_description') else 'N/A',
                'is_remote': job.get('job_is_remote', False),
                'employer_logo': job.get('employer_logo', '')
            })
        
        return jsonify({
            'success': True,
            'query': search_query,
            'total_found': len(result.get('data', [])),
            'jobs': jobs
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False, 'jobs': []}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'name': 'Career Counselor RAG API',
        'status': 'running',
        'endpoints': [
            'POST /search - Search career database',
            'POST /extract-resume - Extract skills from PDF',
            'POST /analyze-skills - Compare skills with job requirements',
            'POST /search-jobs - Search real job listings',
            'GET /health - Health check'
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "="*50)
    print("Career Guidance RAG API (Lite)")
    print("="*50)
    print("Endpoints:")
    print("  POST /search         - Search career database")
    print("  POST /extract-resume - Extract skills from PDF")
    print("  POST /analyze-skills - Compare skills with job requirements")
    print("  POST /search-jobs    - Search REAL job listings")
    print("  GET  /health         - Health check")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
