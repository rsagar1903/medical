import streamlit as st
import pandas as pd
import torch
import chromadb
import requests
import json
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
import time
from datetime import datetime
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import hashlib
from functools import lru_cache
import httpx
from groq import Groq  # Added for Cloud Deployment

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from pypdf import PdfReader, PdfWriter
from docx import Document

# --- Helper Functions (Safe to be at global scope) ---

def extract_template_outline(template_bytes: bytes) -> List[str]:
    """Module-level extractor for PDF template headings to avoid class reload ordering issues."""
    try:
        reader = PdfReader(BytesIO(template_bytes))
        text = []
        for page in reader.pages[:3]:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                continue
        joined = "\n".join(text)
        lines = [l.strip() for l in joined.splitlines()]
        candidates: List[str] = []
        for line in lines:
            if not line:
                continue
            if len(line) < 3 or len(line) > 80:
                continue
            if line.lower().startswith("page "):
                continue
            looks_like_heading = (
                line.endswith(":") or
                (line.isupper() and any(c.isalpha() for c in line)) or
                (line.istitle() and sum(ch.isalpha() for ch in line) >= 6)
            )
            if looks_like_heading:
                normalized = line.rstrip(":").strip()
                if normalized not in candidates:
                    candidates.append(normalized)
        return candidates[:30] if candidates else []
    except Exception:
        return []

# Try to import autogen, but make it optional
try:
    import pyautogen
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Note: st.warning is moved inside main()

# Performance: HTTP session and model/db connectors
def _get_css(minimal: bool) -> str:
    if minimal:
        return """
<style>
    :root {
        --bg:#ffffff; 
        --fg:#0f172a; 
        --muted:#64748b; 
        --border:#e2e8f0; 
        --primary:#0ea5e9; 
        --primary-dark:#0284c7;
        --accent-blue:#0ea5e9;
        --text-dark:#0f172a;
        --text-light:#64748b;
        --success-green:#16a34a;
        --danger-red:#dc2626;
        --card:#ffffff;
        --card-muted:#f8fafc;
    }
    .main-header { padding: 1rem; border: 1px solid var(--border); border-radius: 12px; background: var(--card); color: var(--fg); }
    .main-header h1 { margin:0; font-size: 1.4rem; }
    .patient-card, .metric-card, .chat-container, .summary-card { border: 1px solid var(--border); border-radius: 12px; padding: 1rem; background: var(--card); color: var(--fg); }
    .patient-card h4, .metric-card h4, .summary-card h3 { color: var(--text-dark); }
    .chat-message { border:1px solid var(--border); border-left:4px solid var(--primary); border-radius:10px; padding:.75rem; background:var(--card-muted); color: var(--text-dark); }
    .doctor-message { background:var(--card-muted); }
    .ai-message { background:var(--card-muted); border-left-color:#9333ea; }
    .stButton > button { background: var(--primary); color:#fff; border:0; border-radius:10px; padding:.6rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.05); }
    .stButton > button:hover { background: var(--primary-dark); }
    .stTextArea textarea, .stTextInput input { border-radius:10px !important; border:1px solid var(--border) !import; }
</style>
"""
    return ""

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests import Session

def _http_session() -> Session:
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.headers.update({"Connection": "keep-alive"})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Embedding cache to avoid recomputing embeddings
_embedding_cache = {}

def _get_text_hash(text: str) -> str:
    """Generate hash for text to use as cache key"""
    return hashlib.md5(text.encode()).hexdigest()

# FastAPI client configuration
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

class FastAPIClient:
    """Client for FastAPI backend"""
    def __init__(self, base_url: str = FASTAPI_BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)
    
    def chat(self, message: str, patient_data: Optional[Dict] = None) -> str:
        """Chat with AI agent via FastAPI"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json={"message": message, "patient_data": patient_data}
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.RequestError as e:
            return f"❌ Error connecting to API: {str(e)}"
        except httpx.HTTPStatusError as e:
            return f"❌ API error: {e.response.text}"
    
    def generate_summary(self, patient_data: str, template_outline: Optional[List[str]] = None) -> str:
        """Generate discharge summary via FastAPI"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate-summary",
                json={"patient_data": patient_data, "template_outline": template_outline}
            )
            response.raise_for_status()
            return response.json()["summary"]
        except httpx.RequestError as e:
            return f"❌ Error connecting to API: {str(e)}"
        except httpx.HTTPStatusError as e:
            return f"❌ API error: {e.response.text}"
    
    def search_similar(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Search similar cases via FastAPI"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/search-similar",
                json={"query_text": query_text, "n_results": n_results}
            )
            response.raise_for_status()
            return response.json()["similar_cases"]
        except httpx.RequestError as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            return []
        except httpx.HTTPStatusError as e:
            st.error(f"❌ API error: {e.response.text}")
            return []
    
    def get_patient(self, unit_no: str) -> Optional[Dict]:
        """Get patient via FastAPI"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/patient",
                json={"unit_no": unit_no}
            )
            response.raise_for_status()
            return response.json()["patient"]
        except httpx.RequestError as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            st.error(f"❌ API error: {e.response.text}")
            return None
    
    def health_check(self) -> bool:
        """Check if FastAPI backend is available"""
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()

def _load_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

def _connect_mongo(uri: str):
    client = MongoClient(uri)
    return client

def _connect_chroma(path: str):
    # IMPORTANT for Cloud: Check if folder exists, otherwise recreate or handle error
    if not os.path.exists(path):
        print(f"⚠️ Warning: ChromaDB path '{path}' not found. Ensure you pushed 'vector_db' to Git.")
    
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection("patient_embeddings")
    return client, collection

# Page configuration
st.set_page_config(
    page_title="Medical Discharge Summary Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Modern Color Palette */
    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #141b2d;
        --bg-tertiary: #1a2332;
        --bg-card: #1e293b;
        --bg-card-hover: #243447;
        --bg-input: #0f172a;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #334155;
        --border-light: #475569;
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --primary-light: #818cf8;
        --accent-purple: #a855f7;
        --accent-cyan: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        --gradient-secondary: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --gradient-card: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styles */
    .main-header {
        padding: 3rem 2.5rem;
        border: none;
        border-radius: 24px;
        background: var(--gradient-primary);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styles */
    .patient-card, .metric-card, .chat-container, .summary-card {
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        background: var(--gradient-card);
        color: var(--text-primary);
        box-shadow: var(--shadow-md);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .patient-card:hover, .metric-card:hover, .summary-card:hover {
        box-shadow: var(--shadow-lg), 0 0 30px rgba(99, 102, 241, 0.2);
        transform: translateY(-4px);
        border-color: var(--primary);
    }
    
    .metric-card {
        text-align: center;
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .metric-card h4 {
        margin: 0 0 0.75rem 0;
        color: var(--primary-light);
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .metric-card p {
        margin: 0.5rem 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Chat Message Styles */
    .chat-message {
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary);
        border-radius: 16px;
        padding: 1.25rem;
        background: var(--bg-card);
        color: var(--text-primary);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInUp 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .doctor-message {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-left-color: var(--primary);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .ai-message {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(236, 72, 153, 0.1) 100%);
        border-left-color: var(--accent-purple);
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* Button Styles */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: var(--shadow-md) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.01em !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-hover) !important;
        box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Input Styles */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
        border: 2px solid var(--border-color) !important;
        padding: 0.875rem 1rem !important;
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Empty State */
    .empty-state {
        width: 100%;
        text-align: center;
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 4rem 2rem;
        background: var(--bg-card);
        color: var(--text-primary);
    }
    
    .empty-state .icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        opacity: 0.7;
        filter: drop-shadow(0 4px 8px rgba(99, 102, 241, 0.3));
    }
    
    .empty-state h3 {
        margin: 0 0 0.75rem 0;
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .empty-state p {
        margin: 0;
        color: var(--text-muted);
        font-size: 1rem;
    }
    
    /* Chat Container */
    .chat-container {
        max-height: 550px;
        overflow-y: auto;
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 5px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
        transition: background 0.3s ease;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Summary Card */
    .summary-card {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.6;
        }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Card Header */
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .card-header-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: var(--gradient-primary);
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: var(--bg-secondary);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: var(--gradient-primary);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--success);
        border-radius: 8px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--danger);
        border-radius: 8px;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Text Area for Summary */
    .stTextArea textarea {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background: var(--bg-input);
        border-color: var(--border-color);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(6, 182, 212, 0.1);
        border-left: 4px solid var(--accent-cyan);
        border-radius: 8px;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Main content text */
    .main .stMarkdown p,
    .main .stMarkdown li {
        color: var(--text-secondary) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: var(--bg-input) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Form */
    .stForm {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        background: var(--bg-card) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'discharge_summary' not in st.session_state:
    st.session_state.discharge_summary = None
if 'autogen_agent' not in st.session_state:
    st.session_state.autogen_agent = None

class MedicalRAGSystem:
    def __init__(self):
        self.mongo_uri = "mongodb+srv://ishaanroopesh0102:6eShFuC0pNnFFNGm@cluster0.biujjg4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.chroma_path = "vector_db/chroma"
        self.ollama_model = "llama3"
        self.num_results = 3
        self.http = _http_session()
        self.embedding_cache = _embedding_cache
        
        # --- CLOUD DEPLOYMENT LOGIC ---
        # Try to get Groq API key from Streamlit Secrets
        try:
            if "GROQ_API_KEY" in st.secrets:
                self.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                self.use_cloud_api = True
                print("✅ Using Groq Cloud API for Inference")
            else:
                print("⚠️ GROQ_API_KEY not found in secrets. Using Local Ollama for Inference.")
                self.use_cloud_api = False
        except Exception:
            self.use_cloud_api = False
            print("⚠️ Using Local Ollama for Inference (Secrets not found)")
        
        # Initialize models
        self._load_models()
        self._connect_databases()

    def extract_template_outline(self, template_bytes: bytes) -> list[str]:
        """Extract an ordered list of section headings from a PDF template."""
        try:
            reader = PdfReader(BytesIO(template_bytes))
            text = []
            for page in reader.pages[:3]:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
            joined = "\n".join(text)
            lines = [l.strip() for l in joined.splitlines()]
            candidates: list[str] = []
            for line in lines:
                if not line:
                    continue
                if len(line) < 3 or len(line) > 80:
                    continue
                if line.lower().startswith("page "):
                    continue
                looks_like_heading = (
                    line.endswith(":") or
                    (line.isupper() and any(c.isalpha() for c in line)) or
                    (line.istitle() and sum(ch.isalpha() for ch in line) >= 6)
                )
                if looks_like_heading:
                    normalized = line.rstrip(":").strip()
                    if normalized not in candidates:
                        candidates.append(normalized)
            return candidates[:30] if candidates else []
        except Exception:
            return []

    def generate_discharge_summary_with_template(self, patient_data: str, outline_sections: list[str]) -> str:
        """Generate discharge summary following the provided ordered outline sections."""
        outline_bullets = "\n".join([f"- {s}" for s in outline_sections])
        system_prompt = f"""You are an expert medical AI assistant that generates a clinically accurate discharge summary.
Follow the section order EXACTLY as specified by the provided outline. Do not add extra sections; if information is missing, write "[Information not available]".

REQUIRED SECTION ORDER (USE EXACT TITLES):
{outline_bullets}

Rules:
- Use concise, professional medical language.
- Base content solely on the input patient data.
- Preserve patient identifiers verbatim if present.
- Be brief and factual."""

        user_prompt = f"""Generate a discharge summary STRICTLY following the section list above, based only on this data:\n\n{patient_data}\n\nReturn plain text with the exact section headings in order."""

        # --- CLOUD VS LOCAL LOGIC ---
        if self.use_cloud_api:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1500,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"❌ Error generating summary via Cloud: {str(e)}"
        else:
            # Local Ollama Fallback
            try:
                response = self.http.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": True,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.85,
                            "max_tokens": 500,
                            "num_predict": 500
                        }
                    },
                    timeout=30
                )
                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            json_data = json.loads(line)
                            if 'message' in json_data and 'content' in json_data['message']:
                                content = json_data['message']['content']
                                if content:
                                    full_response += content
                            if json_data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip() if full_response.strip() else "Summary generated successfully."
                else:
                    return f"❌ Error generating summary: {response.text}"
            except Exception as e:
                return f"❌ Error connecting to Ollama: {str(e)}"

    def generate_pdf_from_text(self, text: str, template_bytes: bytes | None = None) -> bytes:
        """Generate a PDF from plain text."""
        # Determine page size
        page_size = A4
        template_reader = None
        if template_bytes:
            try:
                template_reader = PdfReader(BytesIO(template_bytes))
                first_page = template_reader.pages[0]
                width = float(first_page.mediabox.width)
                height = float(first_page.mediabox.height)
                page_size = (width, height)
            except Exception:
                template_reader = None

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=page_size, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
        )
        story = []
        for para in text.split("\n\n"):
            story.append(Paragraph(para.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 0.18 * inch))
        doc.build(story)
        generated_pdf_bytes = buf.getvalue()

        if not template_reader:
            return generated_pdf_bytes

        gen_reader = PdfReader(BytesIO(generated_pdf_bytes))
        writer = PdfWriter()
        num_pages = max(len(template_reader.pages), len(gen_reader.pages))

        for i in range(num_pages):
            template_page = None
            if i < len(template_reader.pages):
                template_page = template_reader.pages[i]

            if i < len(gen_reader.pages):
                gen_page = gen_reader.pages[i]
                if template_page is not None:
                    try:
                        template_page.merge_page(gen_page)
                        writer.add_page(template_page)
                    except Exception:
                        writer.add_page(gen_page)
                else:
                    writer.add_page(gen_page)
            else:
                if template_page is not None:
                    writer.add_page(template_page)

        out_buf = BytesIO()
        writer.write(out_buf)
        return out_buf.getvalue()

    def generate_docx_from_text(self, text: str) -> bytes:
        """Generate a DOCX file from plain text."""
        doc = Document()
        lines = text.split("\n")
        if lines and len(lines[0]) <= 120 and any(ch.isalpha() for ch in lines[0]):
            doc.add_heading(lines[0].strip(), level=1)
            text = "\n".join(lines[1:])
        for block in text.split("\n\n"):
            for ln in block.split("\n"):
                doc.add_paragraph(ln)
            doc.add_paragraph("")
        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()
    
    def _load_models(self):
        """Load Bio ClinicalBERT model for embeddings"""
        with st.spinner("Loading Bio ClinicalBERT model..."):
            self.tokenizer, self.model = _load_tokenizer_model()
    
    def _connect_databases(self):
        """Connect to MongoDB and ChromaDB"""
        try:
            # MongoDB connection
            self.mongo_client = _connect_mongo(self.mongo_uri)
            self.db = self.mongo_client["hospital_db"]
            self.patients_collection = self.db["test_patients"]
            
            # ChromaDB connection
            self.chroma_client, self.chroma_collection = _connect_chroma(self.chroma_path)
            
            st.success("✅ Connected to databases successfully")
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Bio ClinicalBERT with caching"""
        text_hash = _get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            emb = cls_embedding.squeeze(0)
            if emb.is_cuda:
                emb = emb.to("cpu")
            embedding = emb.tolist()
        
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    def format_patient_fields(self, record: Dict) -> str:
        """Format patient record fields for embedding"""
        fields = [
            "name", "unit no", "admission date", "date of birth", "sex", "service",
            "allergies", "attending", "chief complaint", "major surgical or invasive procedure",
            "history of present illness", "past medical history", "social history",
            "family history", "physical exam", "pertinent results", "medications on admission",
            "brief hospital course", "discharge medications", "discharge diagnosis",
            "discharge condition", "discharge instructions", "follow-up", "discharge disposition"
        ]
        parts = [f"{field.title()}: {record.get(field, '')}" for field in fields if record.get(field)]
        return " ".join(parts)
    
    def get_patient_by_unit_no(self, unit_no: str) -> Optional[Dict]:
        """Retrieve patient record from MongoDB"""
        try:
            record = self.patients_collection.find_one({"unit no": int(unit_no)})
            return record
        except Exception as e:
            st.error(f"Error retrieving patient: {str(e)}")
            return None
    
    def search_similar_cases(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Search for similar cases using RAG"""
        try:
            query_embedding = self.embed_text(query_text)
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            
            similar_cases = []
            for i in range(len(results["documents"][0])):
                similar_cases.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            return similar_cases
        except Exception as e:
            st.error(f"Error searching similar cases: {str(e)}")
            return []
    
    def generate_discharge_summary(self, patient_data: str, similar_cases: List[Dict] = None) -> str:
        """Generate discharge summary using Ollama LLM"""
        system_prompt = """You are an expert medical AI assistant that generates structured, clinically accurate discharge summaries.
Base your summary entirely on the INPUT PATIENT DATA provided.
The discharge summary MUST include: Name, Unit No, Date Of Birth, Sex, Admission/Discharge Dates, Attending, Chief Complaint, Procedure, History, Physical Exam (on Admission), Pertinent Results, Brief Hospital Course, Medications on Admission, Discharge Medications, Discharge Instructions, Discharge Disposition, Discharge Diagnosis, Discharge Condition, Follow-up.

For Name, Unit No, Date of Birth, and Sex, copy the information verbatim.
If information is missing, state "[Information not available]".
Use concise, professional medical language. Be brief and factual."""

        user_prompt = f"""Generate a discharge summary for this patient:
{patient_data}

Extract Name, Unit No, Date of Birth, and Sex exactly as provided."""

        # --- CLOUD VS LOCAL LOGIC ---
        if self.use_cloud_api:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=500,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"❌ Error generating summary via Cloud: {str(e)}"
        else:
            # Local Ollama Fallback
            try:
                response = self.http.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": True,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.85,
                            "max_tokens": 500,
                            "num_predict": 500
                        }
                    },
                    timeout=30
                )

                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            json_data = json.loads(line)
                            if 'message' in json_data and 'content' in json_data['message']:
                                content = json_data['message']['content']
                                if content:
                                    full_response += content
                            if json_data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip() if full_response.strip() else "Summary generated successfully."
                else:
                    return f"❌ Error generating summary: {response.text}"
            except Exception as e:
                return f"❌ Error connecting to Ollama: {str(e)}"

    def add_summary_to_vector_db(self, patient_info: Dict, summary_text: str):
        """
        Embeds the finalized discharge summary and adds it to the ChromaDB collection.
        This serves as the feedback loop, adding a high-quality, human-reviewed
        document back into the RAG system.
        """
        if not summary_text or not patient_info:
            st.warning("No summary text or patient info available to add.")
            return False

        unit_no = patient_info.get('unit no', 'unknown')
        patient_name = patient_info.get('name', 'Unknown')
        
        try:
            # 1. Generate embedding for the new summary
            summary_embedding = self.embed_text(summary_text)
            
            # 2. Prepare a unique ID
            doc_id = f"summary_{unit_no}_{int(time.time())}"
            
            # 3. Prepare metadata
            metadata = {
                "unit_no": str(unit_no),
                "name": patient_name,
                "summary": summary_text[:500],
                "source_type": "feedback_summary"
            }
            
            # 4. Add to ChromaDB
            self.chroma_collection.add(
                embeddings=[summary_embedding],
                documents=[summary_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            try:
                st.toast(f"Database updated: Summary for {unit_no} added.", icon="✅")
            except Exception:
                st.success(f"Database updated: Summary for {unit_no} added.")
            return True
        
        except Exception as e:
            st.error(f"❌ Error adding feedback summary to vector DB: {str(e)}")
            st.exception(e)
            return False
    
class AutoGenMedicalAgent:
    def __init__(self, rag_system: MedicalRAGSystem, api_client: FastAPIClient = None):
        self.rag_system = rag_system
        self.api_client = api_client
        self.agent = None
        self.user_proxy = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize AutoGen medical assistant agent"""
        pass
    
    def chat_with_doctor(self, message: str, patient_data: Dict = None) -> str:
        """Handle conversation with doctor - uses FastAPI if available"""
        try:
            # Use FastAPI if available
            if self.api_client and self.api_client.health_check():
                return self.api_client.chat(message, patient_data)
            else:
                return self._fallback_chat(message, patient_data)
        except Exception as e:
            return f"❌ Error in conversation: {str(e)}"
    
    def _fallback_chat(self, message: str, patient_data: Dict = None) -> str:
        """Fallback chat using direct Ollama/Groq interaction"""
        try:
            # Check if user is asking for discharge summary generation
            if "discharge summary" in message.lower() or "generate summary" in message.lower():
                if patient_data:
                    patient_text = self.rag_system.format_patient_fields(patient_data)
                    return self.rag_system.generate_discharge_summary(patient_text)
                else:
                    return "❌ Please select a patient first to generate a discharge summary."
            
            context = ""
            if patient_data:
                # Truncate context to avoid long prompts
                context = f"\n\nPatient: {patient_data.get('name', 'Unknown')} (Unit {patient_data.get('unit no', 'N/A')})"
            
            system_prompt = """You are a medical AI assistant. Provide concise, accurate responses. Keep answers brief (2-3 sentences max)."""
            
            # --- CLOUD VS LOCAL LOGIC ---
            if self.rag_system.use_cloud_api:
                chat_completion = self.rag_system.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{message}{context}"}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.6,
                    max_tokens=250,
                )
                return chat_completion.choices[0].message.content
            else:
                # Local Fallback
                response = self.rag_system.http.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{message}{context}"}
                        ],
                        "stream": True,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.85,
                            "max_tokens": 150,
                            "num_predict": 150
                        }
                    },
                    timeout=20
                )
                
                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                json_data = json.loads(line)
                                if 'message' in json_data and 'content' in json_data['message']:
                                    content = json_data['message']['content']
                                    if content:
                                        full_response += content
                                if json_data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response.strip() if full_response.strip() else "I'm here to help with medical questions. How can I assist you?"
                else:
                    return f"❌ Error connecting to Ollama: {response.text}"
                
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Please try again with a shorter message."
        except Exception as e:
            return f"❌ Error in fallback chat: {str(e)}"

def main():
    # Initialize FastAPI client
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FastAPIClient()
    
    # Check FastAPI availability
    use_fastapi = st.session_state.api_client.health_check()
    
    # Force fallback if on Cloud (API won't be running)
    # We assume cloud if GROQ_API_KEY is present in secrets
    is_cloud_env = "GROQ_API_KEY" in st.secrets
    
    if use_fastapi and not is_cloud_env:
        st.session_state.use_fastapi = True
        # Initialize with FastAPI client
        if 'rag_system' not in st.session_state:
            try:
                st.session_state.rag_system = MedicalRAGSystem()
            except Exception as e:
                st.warning(f"⚠️ Could not initialize full RAG system: {str(e)}. Some features may be limited.")
        
        if 'autogen_agent' not in st.session_state:
            try:
                st.session_state.autogen_agent = AutoGenMedicalAgent(
                    st.session_state.rag_system if 'rag_system' in st.session_state else None,
                    st.session_state.api_client
                )
            except Exception as e:
                st.error(f"❌ Failed to initialize AI agent: {str(e)}")
    else:
        st.session_state.use_fastapi = False
        # Initialize RAG system as fallback (Cloud mode will land here)
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing Medical RAG System..."):
                try:
                    st.session_state.rag_system = MedicalRAGSystem()
                    st.session_state.autogen_agent = AutoGenMedicalAgent(st.session_state.rag_system, None)
                    
                    # Only show warning if NOT in cloud mode (since cloud expects fallback)
                    if not is_cloud_env:
                        st.warning("""
                        ⚠️ **FastAPI backend not available. Using fallback mode.**
                        Run `python start_api.py` locally for better performance.
                        """)
                    else:
                        st.success("✅ Cloud Mode Active: Using Groq API + Local Fallback Logic")
                        
                except Exception as e:
                    st.error(f"❌ Failed to initialize system: {str(e)}")
                    st.stop()
                    
        if 'autogen_agent' not in st.session_state:
            try:
                st.session_state.autogen_agent = AutoGenMedicalAgent(
                    st.session_state.rag_system if 'rag_system' in st.session_state else None,
                    None
                )
            except Exception as e:
                st.error(f"❌ Failed to initialize AI agent: {str(e)}")

    # Sidebar preferences and CSS
    with st.sidebar:
        st.header("⚙️ Preferences")
        minimal_ui = st.checkbox("Minimal UI", value=st.session_state.get('minimal_ui', False))
        st.session_state.minimal_ui = minimal_ui

        st.markdown("---")
        st.header("📎 Insurance Template")
        template_file = st.file_uploader("Upload PDF template (optional)", type=["pdf"], accept_multiple_files=False)
        if template_file is not None:
            st.session_state["template_pdf_bytes"] = template_file.read()
            outline = extract_template_outline(st.session_state["template_pdf_bytes"])
            if outline:
                st.session_state["template_outline"] = outline
                st.success("Template loaded. Outline detected and will be used for generation.")
                with st.expander("Detected Section Order"):
                    for s in outline:
                        st.write(f"• {s}")
            else:
                st.session_state.pop("template_outline", None)
                st.warning("Template loaded but no clear section outline was detected. Will generate standard summary.")
        elif "template_pdf_bytes" not in st.session_state:
            st.info("No template uploaded. Summaries will be generated as plain text or basic PDF.")

    st.markdown(_get_css(st.session_state.get('minimal_ui', False)), unsafe_allow_html=True)

    # Header with modern dark design
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Discharge Summary Assistant</h1>
        <p>AI-Powered Clinical Documentation with RAG and AutoGen Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators with modern cards
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        st.markdown("""
        <div class="metric-card">
            <h4>🟢 System Ready</h4>
            <p>All systems operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        mode_text = "Cloud API" if st.session_state.rag_system.use_cloud_api else "Local Mode"
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ {mode_text}</h4>
            <p>Optimized responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status3:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 AI Active</h4>
            <p>LLaMA 3 + RAG</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status4:
        patient_count = "1" if st.session_state.current_patient else "0"
        st.markdown(f"""
        <div class="metric-card">
            <h4>👤 Patient</h4>
            <p>{patient_count} selected</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for patient search
    with st.sidebar:
        st.header("🔍 Patient Search")
        
        # Patient search form
        with st.form("patient_search"):
            unit_no = st.text_input("Unit Number", placeholder="Enter patient unit number")
            search_button = st.form_submit_button("🔍 Search Patient")
            
            if search_button and unit_no:
                with st.spinner("Searching for patient..."):
                    try:
                        # Use FastAPI if available AND not in cloud mode
                        if st.session_state.get('use_fastapi', False):
                            patient = st.session_state.api_client.get_patient(unit_no)
                        else:
                            patient = st.session_state.rag_system.get_patient_by_unit_no(unit_no)
                        if patient:
                            st.session_state.current_patient = patient
                            st.markdown(f"""
                            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                                <p style="color: var(--success); margin: 0; font-weight: 600;">✅ Found patient: {patient.get('name', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                                <p style="color: var(--danger); margin: 0; font-weight: 600;">❌ Patient not found</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                            <p style="color: var(--danger); margin: 0; font-weight: 600;">❌ Error: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display current patient info
        if st.session_state.current_patient:
            st.markdown("### 👤 Current Patient")
            patient = st.session_state.current_patient
            
            st.markdown(f"""
            <div class="patient-card">
                <h4 style="color: var(--text-primary); margin-bottom: 1.5rem; font-size: 1.3rem; font-weight: 700;">📋 {patient.get('name', 'Unknown')}</h4>
                <p style="color: var(--text-secondary); margin: 0.75rem 0; font-size: 0.95rem;"><strong style="color: var(--primary-light);">Unit No:</strong> {patient.get('unit no', 'N/A')}</p>
                <p style="color: var(--text-secondary); margin: 0.75rem 0; font-size: 0.95rem;"><strong style="color: var(--primary-light);">DOB:</strong> {patient.get('date of birth', 'N/A')}</p>
                <p style="color: var(--text-secondary); margin: 0.75rem 0; font-size: 0.95rem;"><strong style="color: var(--primary-light);">Sex:</strong> {patient.get('sex', 'N/A')}</p>
                <p style="color: var(--text-secondary); margin: 0.75rem 0; font-size: 0.95rem;"><strong style="color: var(--primary-light);">Service:</strong> {patient.get('service', 'N/A')}</p>
                <p style="color: var(--text-secondary); margin: 0.75rem 0; font-size: 0.95rem;"><strong style="color: var(--primary-light);">Attending:</strong> {patient.get('attending', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card-header">
            <div class="card-header-icon">💬</div>
            <h3 style="margin: 0; color: var(--text-dark);">AI Medical Assistant</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat interface
        if st.session_state.current_patient:
            # Chat container with better styling
            st.markdown("""
            <div class="chat-container">
            """, unsafe_allow_html=True)
            
            # Display chat history
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "doctor":
                        st.markdown(f"""
                        <div class="chat-message doctor-message">
                            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                                <div style="background: var(--gradient-primary); color: white; border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; box-shadow: var(--shadow-md);">👨‍⚕️</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 700; color: var(--primary-light); margin-bottom: 0.5rem; font-size: 0.9rem; letter-spacing: 0.02em;">Doctor</div>
                                    <div style="color: var(--text-primary); line-height: 1.7; font-size: 0.95rem;">{message["content"]}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message ai-message">
                            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                                <div style="background: var(--gradient-primary); color: white; border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; box-shadow: var(--shadow-md);">🤖</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 700; color: var(--accent-purple); margin-bottom: 0.5rem; font-size: 0.9rem; letter-spacing: 0.02em;">AI Assistant</div>
                                    <div style="color: var(--text-primary); line-height: 1.7; font-size: 0.95rem;">{message["content"]}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; color: var(--text-muted); padding: 3rem 2rem; font-style: italic; font-size: 1.1rem;">
                    👋 Start a conversation with the AI assistant...
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Chat input form
            with st.form("chat_form", clear_on_submit=False):
                st.markdown("**<span style='color: var(--text-dark); font-weight: 600;'>Ask the AI assistant:</span>**", unsafe_allow_html=True)
                user_message = st.text_area(
                    "message_input", 
                    placeholder="e.g., Generate a discharge summary for this patient",
                    height=100,
                    label_visibility="collapsed"
                )
                
                col_send, col_clear = st.columns([1, 1])
                with col_send:
                    send_button = st.form_submit_button("💬 Send Message", type="primary", use_container_width=True)
                with col_clear:
                    clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
                
                if send_button and user_message.strip():
                    # Add doctor message to history
                    st.session_state.chat_history.append({
                        "role": "doctor",
                        "content": user_message.strip(),
                        "timestamp": datetime.now()
                    })
                    
                    # Get AI response with progress indicator
                    with st.spinner("🤖 AI is thinking..."):
                        try:
                            start_time = time.time()
                            ai_response = st.session_state.autogen_agent.chat_with_doctor(
                                user_message.strip(), 
                                st.session_state.current_patient
                            )
                            elapsed = time.time() - start_time
                            if elapsed < 2:
                                st.success(f"⚡ Response generated in {elapsed:.1f}s")
                            
                            # Add AI response to history
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "content": ai_response,
                                "timestamp": datetime.now()
                            })
                        except Exception as e:
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "content": f"❌ Error: {str(e)}",
                                "timestamp": datetime.now()
                            })
                    
                    st.rerun()
                
                if clear_button:
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Action buttons with modern styling
            st.markdown("---")
            st.markdown("""
            <div class="card-header">
                <div class="card-header-icon">🚀</div>
                <h3 style="margin: 0; color: var(--text-dark);">Quick Actions</h3>
            </div>
            """, unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("📝 Generate Summary", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        status_text.text("📝 Formatting patient data...")
                        progress_bar.progress(20)
                        patient_text = st.session_state.rag_system.format_patient_fields(st.session_state.current_patient)
                        
                        status_text.text("🤖 Generating summary with AI...")
                        progress_bar.progress(40)
                        start_time = time.time()
                        
                        # Use FastAPI if available AND NOT Cloud
                        if st.session_state.get('use_fastapi', False):
                            template_outline = st.session_state.get("template_outline")
                            summary = st.session_state.api_client.generate_summary(patient_text, template_outline)
                        else:
                            # Fallback / Cloud logic
                            if "template_outline" in st.session_state and st.session_state.template_outline:
                                summary = st.session_state.rag_system.generate_discharge_summary_with_template(patient_text, st.session_state.template_outline)
                            else:
                                summary = st.session_state.rag_system.generate_discharge_summary(patient_text)
                        
                        elapsed = time.time() - start_time
                        progress_bar.progress(80)
                        status_text.text("📄 Preparing document...")
                        
                        st.session_state.discharge_summary = summary
                        # Build PDF
                        template_bytes = st.session_state.get("template_pdf_bytes", None)
                        pdf_bytes = st.session_state.rag_system.generate_pdf_from_text(summary, template_bytes=None if st.session_state.get("template_outline") else template_bytes)
                        st.session_state.discharge_summary_pdf = pdf_bytes
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"✅ Discharge summary generated in {elapsed:.1f}s!")
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                    st.rerun()
            
            with col_btn2:
                if st.button("🔍 Find Similar Cases", use_container_width=True):
                    with st.spinner("🔍 Searching for similar cases..."):
                        try:
                            patient_text = st.session_state.rag_system.format_patient_fields(st.session_state.current_patient)
                            # Use FastAPI if available AND NOT Cloud
                            if st.session_state.get('use_fastapi', False):
                                similar_cases = st.session_state.api_client.search_similar(patient_text)
                            else:
                                similar_cases = st.session_state.rag_system.search_similar_cases(patient_text)
                            st.session_state.similar_cases = similar_cases
                            st.success(f"✅ Found {len(similar_cases)} similar cases!")
                        except Exception as e:
                            st.error(f"❌ Error searching cases: {str(e)}")
                        st.rerun()
            
            with col_btn3:
                if st.button("📊 Patient Overview", use_container_width=True):
                    with st.spinner("📊 Analyzing patient data..."):
                        try:
                            patient = st.session_state.current_patient
                            overview = f"""**Patient Overview:**

**Name:** {patient.get('name', 'Unknown')}
**Unit No:** {patient.get('unit no', 'N/A')}
**Date of Birth:** {patient.get('date of birth', 'N/A')}
**Sex:** {patient.get('sex', 'N/A')}
**Service:** {patient.get('service', 'N/A')}
**Chief Complaint:** {patient.get('chief complaint', 'N/A')}
**Attending:** {patient.get('attending', 'N/A')}
**Allergies:** {patient.get('allergies', 'N/A')}
**Past Medical History:** {patient.get('past medical history', 'N/A')[:200]}{'...' if len(str(patient.get('past medical history', ''))) > 200 else ''}

This patient is ready for discharge summary generation."""
                            
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "content": overview,
                                "timestamp": datetime.now()
                            })
                            st.success("✅ Patient overview added to chat!")
                        except Exception as e:
                            st.error(f"❌ Error generating overview: {str(e)}")
                        st.rerun()
        
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">👈</div>
                <h3>Search for a Patient</h3>
                <p>Please search for a patient in the sidebar to start the conversation with the AI assistant.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card-header">
            <div class="card-header-icon">📋</div>
            <h3 style="margin: 0; color: var(--text-dark);">Generated Content</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display discharge summary (editable)
        if st.session_state.discharge_summary:
            st.markdown("""
            <div class="summary-card">
                <div class="card-header">
                    <div class="card-header-icon">📄</div>
                    <h3 style="margin: 0; color: var(--text-dark);">Discharge Summary (Editable)</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "editable_summary" not in st.session_state:
                st.session_state.editable_summary = st.session_state.discharge_summary
            # Use current discharge_summary if editable_summary is empty or reset
            elif not st.session_state.editable_summary:
                st.session_state.editable_summary = st.session_state.discharge_summary

            st.session_state.editable_summary = st.text_area(
                 "editable_summary",
                 value=st.session_state.editable_summary,
                 height=500,
                 label_visibility="collapsed"
             )

            col_save, col_reset = st.columns([1,1])
            with col_save:
                 if st.button("💾 Save Edits", use_container_width=True):
                     st.session_state.discharge_summary = st.session_state.editable_summary
                     st.success("Saved your edits.")
            with col_reset:
                 if st.button("↩️ Reset to Generated", use_container_width=True):
                    st.session_state.editable_summary = st.session_state.discharge_summary
                    st.rerun() # Rerun to ensure text_area updates

            # --- START: NEW FEEDBACK LOOP UI ---
            st.markdown("---")
            st.markdown("### 🧠 RAG Feedback Loop")
            
            if st.button("Commit Summary to Knowledgebase", 
                         type="primary", 
                         use_container_width=True, 
                         help="Embed this summary and add it to the RAG system for future 'similar cases' searches."):
                
                if st.session_state.editable_summary and st.session_state.current_patient:
                    with st.spinner("Embedding summary and updating knowledgebase..."):
                        st.session_state.rag_system.add_summary_to_vector_db(
                            st.session_state.current_patient,
                            st.session_state.editable_summary
                        )
                else:
                    st.warning("Please ensure a patient is loaded and a summary is present.")
            # --- END: NEW FEEDBACK LOOP UI ---
            
            st.markdown("---") # Added a separator

            # Plain text download
            st.download_button(
                label="📥 Download as .txt",
                data=st.session_state.editable_summary,
                file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.txt",
                mime="text/plain"
            )

            # DOCX download
            docx_bytes = st.session_state.rag_system.generate_docx_from_text(st.session_state.editable_summary)
            st.download_button(
                label="📝 Download as .docx",
                data=docx_bytes,
                file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # PDF download (optional template mode)
            if "discharge_summary_pdf" in st.session_state and st.session_state.discharge_summary_pdf:
                st.download_button(
                    label="🧾 Download PDF (Template Applied)",
                    data=st.session_state.discharge_summary_pdf,
                    file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.pdf",
                    mime="application/pdf"
                )
        
        # Display similar cases
        if hasattr(st.session_state, 'similar_cases') and st.session_state.similar_cases:
            st.markdown("### 🔍 Similar Cases Found")
            
            for i, case in enumerate(st.session_state.similar_cases):
                with st.expander(f"Case {i+1} - Similarity: {case['similarity']:.2%}"):
                    st.write("**Patient Info:**")
                    st.write(f"Name: {case['metadata'].get('name', 'Unknown')}")
                    st.write(f"Unit No: {case['metadata'].get('unit_no', 'Unknown')}")
                    
                    st.write("**Summary Preview:**")
                    summary_preview = case['metadata'].get('summary', 'No summary available')[:200] + "..."
                    st.write(summary_preview)

if __name__ == "__main__":
    main()