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
from io import BytesIO

# --- Consolidated Imports (Fixed NameError) ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests import Session  # Explicitly import Session for type hinting
from groq import Groq 
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from pypdf import PdfReader, PdfWriter
from docx import Document

# --- Helper Functions ---

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

# Performance: HTTP session and model/db connectors
def _http_session() -> Session:
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.headers.update({"Connection": "keep-alive"})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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
    return """
<style>
    :root {
        --bg:#ffffff; 
        --fg:#0f172a; 
        --muted:#64748b; 
        --border:#e2e8f0; 
        --primary:#0ea5e9; 
        --primary-dark:#0284c7; 
        --success:#16a34a;
        --accent-blue:#0ea5e9;
        --text-dark:#0f172a;
        --text-light:#64748b;
        --success-green:#16a34a;
        --danger-red:#dc2626;
        --card:#ffffff;
        --card-muted:#f8fafc;
    }
    .main-header { padding: 1rem; border: 1px solid var(--border); border-radius: 12px; background: var(--card); color: var(--fg); text-align:center; margin-bottom: 1rem; }
    .main-header h1 { margin:0; font-size: 1.4rem; }
    .patient-card, .metric-card, .chat-container, .summary-card { border: 1px solid var(--border); border-radius: 12px; padding: 1rem; background: var(--card); color: var(--fg); }
    .metric-card h4 { margin: 0 0 .25rem 0; color: var(--success); font-size: 1rem; }
    .metric-card p { margin: .25rem 0; color: var(--fg); }
    .chat-message { border:1px solid var(--border); border-left:4px solid var(--primary); border-radius:10px; padding:.75rem; background:var(--card-muted); color: var(--text-dark); }
    .doctor-message { background:var(--card-muted); }
    .ai-message { background:var(--card-muted); border-left-color:#9333ea; }
    .stButton > button { background: var(--primary); color:#fff; border:0; border-radius:10px; padding:.6rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.05); }
    .stButton > button:hover { background: var(--primary-dark); }
    .stTextArea textarea, .stTextInput input { border-radius:10px !important; border:1px solid var(--border) !important; }
    .empty-state { width:100%; text-align:center; border: 1px dashed var(--border); border-radius: 12px; padding: 2rem; background:var(--card); color: var(--fg); }
    .empty-state .icon { font-size: 2rem; margin-bottom: .5rem; }
    .empty-state h3 { margin: 0 0 .25rem 0; color: var(--fg); font-size: 1.1rem; font-weight: 600; }
    .empty-state p { margin: 0; color: var(--muted); font-size: .95rem; }
</style>
"""

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
    
    # Use PersistentClient with the exact path
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection("patient_embeddings")
    return client, collection


# --- Business Logic Class ---

class MedicalRAGSystem:
    def __init__(self):
        self.mongo_uri = "mongodb+srv://ishaanroopesh0102:6eShFuC0pNnFFNGm@cluster0.biujjg4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.chroma_path = "vector_db/chroma"
        self.ollama_model = "llama3"
        self.num_results = 3
        self.http = _http_session()
        
        # --- CLOUD DEPLOYMENT LOGIC ---
        # Try to get Groq API key from Streamlit Secrets
        try:
            if "GROQ_API_KEY" in st.secrets:
                self.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                self.use_cloud_api = True
                print("✅ Using Groq Cloud API for Inference")
            else:
                # If key is missing, assume local mode or raise specific error depending on preference
                # Here we fallback to local but print a clear message
                print("⚠️ GROQ_API_KEY not found in secrets. Using Local Ollama for Inference.")
                self.use_cloud_api = False
        except Exception:
            # st.secrets access might fail locally if .streamlit/secrets.toml doesn't exist
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
- Do not invent data; base content solely on the input patient data.
- Preserve patient identifiers verbatim if present.
"""

        user_prompt = f"""Generate a discharge summary STRICTLY following the section list above, based only on this data:\n\n{patient_data}\n\nReturn plain text with the exact section headings in order."""

        # --- CLOUD VS LOCAL LOGIC ---
        if self.use_cloud_api:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-70b-versatile",
                    temperature=0.4,
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
                            "temperature": 0.4,
                            "top_p": 0.9,
                            "max_tokens": 700
                        }
                    },
                    timeout=60
                )
                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            json_data = json.loads(line)
                            if 'message' in json_data and 'content' in json_data['message']:
                                full_response += json_data['message']['content']
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip()
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
        # We check for st.spinner in main() now
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
            
        except Exception as e:
            print(f"Database connection failed: {str(e)}") 
            raise 
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Bio ClinicalBERT"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            emb = cls_embedding.squeeze(0)
            if emb.is_cuda:
                emb = emb.to("cpu")
            return emb.tolist()
    
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
        system_prompt = """You are an expert medical AI assistant tasked with generating a structured, clinically accurate, and concise discharge summary.
Base your summary entirely on the 'INPUT PATIENT DATA' provided.
The discharge summary MUST include all the following sections. For Name, Unit No, Date of Birth, and Sex, you MUST copy the information verbatim.
If essential information for a required section is genuinely absent, state "[Information not available]".

REQUIRED DISCHARGE SUMMARY STRUCTURE:
Name, Unit No, Date Of Birth, Sex, Admission/Discharge Dates, Attending, Chief Complaint, Procedure, History, Physical Exam (on Admission), Pertinent Results, Brief Hospital Course, Medications on Admission, Discharge Medications, Discharge Instructions, Discharge Disposition, Discharge Diagnosis, Discharge Condition, Follow-up.

Maintain a professional, objective medical tone. Do not add conversational phrases."""

        user_prompt = f"""Generate a discharge summary for the following patient based on the provided data:
**INPUT PATIENT DATA (Query):**
{patient_data}

**DISCHARGE SUMMARY (Query):**

**Reminder:** Extract and display the patient's Name, Unit No, Date of Birth, and Sex exactly as provided at the top of the discharge summary. Do not skip or modify them."""

        # --- CLOUD VS LOCAL LOGIC ---
        if self.use_cloud_api:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.1-70b-versatile",
                    temperature=0.4,
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
                        "stream": True
                    },
                    timeout=60
                )

                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            json_data = json.loads(line)
                            if 'message' in json_data and 'content' in json_data['message']:
                                full_response += json_data['message']['content']
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip()
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
    def __init__(self, rag_system: MedicalRAGSystem):
        self.rag_system = rag_system
        self.agent = None
        self.user_proxy = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize AutoGen medical assistant agent"""
        pass
    
    def chat_with_doctor(self, message: str, patient_data: Dict = None) -> str:
        """Handle conversation with doctor"""
        try:
            return self._fallback_chat(message, patient_data)
        except Exception as e:
            return f"❌ Error in conversation: {str(e)}"
    
    def _fallback_chat(self, message: str, patient_data: Dict = None) -> str:
        """Fallback chat using direct Ollama interaction"""
        try:
            if "discharge summary" in message.lower() or "generate summary" in message.lower():
                if patient_data:
                    patient_text = self.rag_system.format_patient_fields(patient_data)
                    return self.rag_system.generate_discharge_summary(patient_text)
                else:
                    return "❌ Please select a patient first to generate a discharge summary."
            
            context = ""
            if patient_data:
                context = f"\n\nCurrent Patient Context:\n{self.rag_system.format_patient_fields(patient_data)}"
            
            system_prompt = """You are a medical AI assistant that helps doctors with discharge summaries and medical questions. 
            Provide helpful, accurate, and professional responses about medical topics. 
            Keep responses concise and focused. If asked about generating a discharge summary, guide the user to use the 'Generate Summary' button."""
            
            # --- CLOUD VS LOCAL LOGIC ---
            if self.rag_system.use_cloud_api:
                chat_completion = self.rag_system.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{message}{context}"}
                    ],
                    model="llama-3.1-70b-versatile",
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
                            "top_p": 0.9,
                            "max_tokens": 250
                        }
                    },
                    timeout=45
                )
                if response.ok:
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                json_data = json.loads(line)
                                if 'message' in json_data and 'content' in json_data['message']:
                                    full_response += json_data['message']['content']
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
    st.set_page_config(
        page_title="Medical Discharge Summary Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if not AUTOGEN_AVAILABLE:
        st.warning("⚠️ AutoGen not available. Some features may be limited.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'discharge_summary' not in st.session_state:
        st.session_state.discharge_summary = None
    if 'autogen_agent' not in st.session_state:
        st.session_state.autogen_agent = None
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing Medical RAG System..."):
            try:
                st.session_state.rag_system = MedicalRAGSystem()
                st.session_state.autogen_agent = AutoGenMedicalAgent(st.session_state.rag_system)
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {str(e)}")
                st.stop()

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

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Discharge Summary Assistant</h1>
        <p>AI-Powered Clinical Documentation with RAG and AutoGen Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for patient search
    with st.sidebar:
        st.header("🔍 Patient Search")
        
        with st.form("patient_search"):
            unit_no = st.text_input("Unit Number", placeholder="Enter patient unit number")
            search_button = st.form_submit_button("🔍 Search Patient")
            
            if search_button and unit_no:
                with st.spinner("Searching for patient..."):
                    try:
                        patient = st.session_state.rag_system.get_patient_by_unit_no(unit_no)
                        if patient:
                            st.session_state.current_patient = patient
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 1px solid var(--success-green); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                                <p style="color: var(--text-dark); margin: 0; font-weight: 600;">✅ Found patient: {patient.get('name', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 1px solid var(--danger-red); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                                <p style="color: var(--text-dark); margin: 0; font-weight: 600;">❌ Patient not found</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 1px solid var(--danger-red); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                            <p style="color: var(--text-dark); margin: 0; font-weight: 600;">❌ Error: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display current patient info
        if st.session_state.current_patient:
            st.markdown("### 👤 Current Patient")
            patient = st.session_state.current_patient
            
            st.markdown(f"""
            <div class="patient-card">
                <h4 style="color: var(--text-dark); margin-bottom: 1rem;">📋 {patient.get('name', 'Unknown')}</h4>
                <p style="color: var(--text-dark); margin: 0.5rem 0;"><strong style="color: var(--accent-blue);">Unit No:</strong> {patient.get('unit no', 'N/A')}</p>
                <p style="color: var(--text-dark); margin: 0.5rem 0;"><strong style="color: var(--accent-blue);">DOB:</strong> {patient.get('date of birth', 'N/A')}</p>
                <p style="color: var(--text-dark); margin: 0.5rem 0;"><strong style="color: var(--accent-blue);">Sex:</strong> {patient.get('sex', 'N/A')}</p>
                <p style="color: var(--text-dark); margin: 0.5rem 0;"><strong style="color: var(--accent-blue);">Service:</strong> {patient.get('service', 'N/A')}</p>
                <p style="color: var(--text-dark); margin: 0.5rem 0;"><strong style="color: var(--accent-blue);">Attending:</strong> {patient.get('attending', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 AI Medical Assistant")
        
        if st.session_state.current_patient:
            st.markdown("""
            <div style="background: #ffffff; border-radius: 15px; padding: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 1rem; max-height: 400px; overflow-y: auto; border: 1px solid #e9ecef;">
            """, unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "doctor":
                        st.markdown(f"""
                        <div class="chat-message doctor-message" style="margin: 1rem 0; text-align: left;">
                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                <div style="background: var(--accent-blue); color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0;">👨‍⚕️</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 600; color: var(--accent-blue); margin-bottom: 0.25rem;">Doctor</div>
                                    <div style="color: var(--text-dark); line-height: 1.5;">{message["content"]}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message ai-message" style="margin: 1rem 0; text-align: left;">
                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                <div style="background: #9c27b0; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0;">🤖</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 600; color: #9c27b0; margin-bottom: 0.25rem;">AI Assistant</div>
                                    <div style="color: var(--text-dark); line-height: 1.5;">{message["content"]}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; color: var(--text-light); padding: 2rem; font-style: italic;">
                    👋 Start a conversation with the AI assistant...
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
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
                    st.session_state.chat_history.append({
                        "role": "doctor",
                        "content": user_message.strip(),
                        "timestamp": datetime.now()
                    })
                    
                    with st.spinner("🤖 AI is thinking..."):
                        try:
                            ai_response = st.session_state.autogen_agent.chat_with_doctor(
                                user_message.strip(), 
                                st.session_state.current_patient
                            )
                            
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
            
            st.markdown("### 🚀 Quick Actions")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("📝 Generate Summary", type="primary", use_container_width=True):
                    with st.spinner("📝 Generating discharge summary..."):
                        try:
                            patient_text = st.session_state.rag_system.format_patient_fields(st.session_state.current_patient)
                            if "template_outline" in st.session_state and st.session_state.template_outline:
                                summary = st.session_state.rag_system.generate_discharge_summary_with_template(patient_text, st.session_state.template_outline)
                            else:
                                summary = st.session_state.rag_system.generate_discharge_summary(patient_text)
                            st.session_state.discharge_summary = summary
                            template_bytes = st.session_state.get("template_pdf_bytes", None)
                            pdf_bytes = st.session_state.rag_system.generate_pdf_from_text(summary, template_bytes=None if st.session_state.get("template_outline") else template_bytes)
                            st.session_state.discharge_summary_pdf = pdf_bytes
                            st.success("✅ Discharge summary generated!")
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {str(e)}")
                        st.rerun()
            
            with col_btn2:
                if st.button("🔍 Find Similar Cases", use_container_width=True):
                    with st.spinner("🔍 Searching for similar cases..."):
                        try:
                            patient_text = st.session_state.rag_system.format_patient_fields(st.session_state.current_patient)
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
        st.header("📋 Generated Content")
        
        if st.session_state.discharge_summary:
            st.markdown("""
            <div class="summary-card">
                <h3>📄 Discharge Summary (Editable)</h3>
            </div>
            """, unsafe_allow_html=True)

            if "editable_summary" not in st.session_state:
                st.session_state.editable_summary = st.session_state.discharge_summary
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
                    st.rerun()

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
            
            st.markdown("---")

            st.download_button(
                label="📥 Download as .txt",
                data=st.session_state.editable_summary,
                file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.txt",
                mime="text/plain"
            )

            docx_bytes = st.session_state.rag_system.generate_docx_from_text(st.session_state.editable_summary)
            st.download_button(
                label="📝 Download as .docx",
                data=docx_bytes,
                file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            if "discharge_summary_pdf" in st.session_state and st.session_state.discharge_summary_pdf:
                st.download_button(
                    label="🧾 Download PDF (Template Applied)",
                    data=st.session_state.discharge_summary_pdf,
                    file_name=f"discharge_summary_{st.session_state.current_patient.get('unit no', 'unknown')}.pdf",
                    mime="application/pdf"
                )
        
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
    
    # Footer with system status
    st.markdown("---")
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.markdown("""
        <div class="metric-card">
            <h4>Database Status</h4>
            <p>🟢 Connected</p>
            <p style="color: var(--muted);">MongoDB + ChromaDB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        st.markdown("""
        <div class="metric-card">
            <h4>AI Model</h4>
            <p>🟢 Ready</p>
            <p style="color: var(--muted);">Bio ClinicalBERT + LLaMA 3</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status3:
        st.markdown("""
        <div class="metric-card">
            <h4>AI Assistant</h4>
            <p>🟢 Active</p>
            <p style="color: var(--muted);">Medical Assistant</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()