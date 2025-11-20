"""
Configuration file for Medical Discharge Summary Assistant
Centralized configuration for all application settings
"""

import os
from pathlib import Path

# --- Database Configuration ---
# Best Practice: Try to get URI from environment first for security
# If not found, fallback to the hardcoded string (for local dev)
MONGO_URI = os.environ.get(
    "MONGO_URI", 
    "mongodb+srv://ishaanroopesh0102:6eShFuC0pNnFFNGm@cluster0.biujjg4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

CHROMA_PATH = "vector_db/chroma"
DATABASE_NAME = "hospital_db"
PATIENTS_COLLECTION = "test_patients"

# --- AI Model Configuration (Local) ---
BIO_CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# --- Cloud Inference Configuration (Groq) ---
# These settings are used when the app detects a GROQ_API_KEY
GROQ_MODEL_ID = "llama3-8b-8192"
GROQ_MAX_TOKENS = 1500
GROQ_TEMPERATURE = 0.4

# --- RAG Configuration ---
NUM_SIMILAR_CASES = 3
EMBEDDING_MAX_LENGTH = 512
SIMILARITY_THRESHOLD = 0.7

# --- AutoGen Configuration ---
AUTOGEN_CONFIG = {
    "model": "llama3",
    "api_base": f"{OLLAMA_BASE_URL}/v1",
    "api_key": "ollama",
    "temperature": 0.7,
    "max_tokens": 2000,
}

# --- UI Configuration ---
APP_TITLE = "Medical Discharge Summary Assistant"
APP_ICON = "🏥"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# --- File Paths ---
DATA_DIR = Path("data")
EMBEDDINGS_DIR = Path("embeddings")
PROCESSED_DIR = Path("processed")
SCRIPTS_DIR = Path("scripts")
VECTOR_DB_DIR = Path("vector_db")

# --- Discharge Summary Template ---
DISCHARGE_SUMMARY_SECTIONS = [
    "Name",
    "Unit No", 
    "Date Of Birth",
    "Sex",
    "Admission/Discharge Dates",
    "Attending",
    "Chief Complaint",
    "Procedure",
    "History",
    "Physical Exam (on Admission)",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Instructions",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Follow-up"
]

# --- System Prompts ---
SYSTEM_PROMPT = """You are an expert medical AI assistant tasked with generating a structured, clinically accurate, and concise discharge summary.
Base your summary entirely on the 'INPUT PATIENT DATA' provided.
The discharge summary MUST include all the following sections. For Name, Unit No, Date of Birth, and Sex, you MUST copy the information verbatim.
If essential information for a required section is genuinely absent, state "[Information not available]".

REQUIRED DISCHARGE SUMMARY STRUCTURE:
Name, Unit No, Date Of Birth, Sex, Admission/Discharge Dates, Attending, Chief Complaint, Procedure, History, Physical Exam (on Admission), Pertinent Results, Brief Hospital Course, Medications on Admission, Discharge Medications, Discharge Instructions, Discharge Disposition, Discharge Diagnosis, Discharge Condition, Follow-up.

Maintain a professional, objective medical tone. Do not add conversational phrases."""

AUTOGEN_SYSTEM_MESSAGE = """You are a specialized medical AI assistant that helps doctors generate discharge summaries. 
You have access to a RAG system that can retrieve similar patient cases and generate comprehensive discharge summaries.

Your capabilities:
1. Retrieve patient information from the hospital database
2. Search for similar cases using medical embeddings
3. Generate structured discharge summaries
4. Answer questions about patient care and medical procedures

Always maintain professional medical terminology and ensure accuracy in all medical information.
When generating discharge summaries, follow the standard medical format and include all required sections."""

# --- Patient Data Fields ---
PATIENT_FIELDS = [
    "name", "unit no", "admission date", "date of birth", "sex", "service",
    "allergies", "attending", "chief complaint", "major surgical or invasive procedure",
    "history of present illness", "past medical history", "social history",
    "family history", "physical exam", "pertinent results", "medications on admission",
    "brief hospital course", "discharge medications", "discharge diagnosis",
    "discharge condition", "discharge instructions", "follow-up", "discharge disposition"
]

# --- Environment Variables ---
ENVIRONMENT_VARIABLES = {
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": str(Path.home() / ".cache" / "transformers"),
    "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
}

# --- Logging Configuration ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Performance Settings ---
BATCH_SIZE = 100
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 30
CACHE_TTL = 3600  # 1 hour

# --- Security Settings ---
ENABLE_CORS = True
ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# --- Feature Flags ---
ENABLE_AUTOGEN = True
ENABLE_RAG_SEARCH = True
ENABLE_SIMILAR_CASES = True
ENABLE_DOWNLOAD = True
ENABLE_CHAT_HISTORY = True

# --- UI Themes ---
THEME_CONFIG = {
    "primaryColor": "#007bff",
    "backgroundColor": "#ffffff", 
    "secondaryBackgroundColor": "#f8f9fa",
    "textColor": "#262730",
    "font": "sans serif"
}

# --- Error Messages ---
ERROR_MESSAGES = {
    "patient_not_found": "❌ Patient not found in database",
    "database_connection": "❌ Failed to connect to database",
    "model_loading": "❌ Failed to load AI model",
    "ollama_connection": "❌ Failed to connect to Ollama service",
    "embedding_generation": "❌ Failed to generate embeddings",
    "summary_generation": "❌ Failed to generate discharge summary",
    "similar_cases_search": "❌ Failed to search similar cases",
    "autogen_initialization": "❌ Failed to initialize AutoGen agent"
}

# --- Success Messages ---
SUCCESS_MESSAGES = {
    "patient_found": "✅ Patient found successfully",
    "database_connected": "✅ Connected to database successfully",
    "model_loaded": "✅ AI model loaded successfully",
    "ollama_connected": "✅ Connected to Ollama service",
    "summary_generated": "✅ Discharge summary generated successfully",
    "similar_cases_found": "✅ Similar cases found successfully",
    "autogen_ready": "✅ AutoGen agent ready"
}