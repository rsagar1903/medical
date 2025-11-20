🏥 Medical Discharge Summary Assistant

AI-Powered Clinical Documentation with RAG, ChromaDB & Llama 3 (Groq Cloud)

This project is a full-stack medical documentation system that helps clinicians automatically generate high-quality discharge summaries using:

BioClinicalBERT for embeddings

ChromaDB for similarity search

MongoDB Atlas for patient records

Groq Cloud (Llama 3) for cloud-based LLM inference

Streamlit for an interactive UI

This deployment is optimized for Streamlit Cloud, which cannot run local models, so we use hybrid cloud architecture with Groq.

🚀 Features
🧠 AI-Generated Discharge Summaries

Fully structured summaries using standard clinical headings

Template-aware generation (if a PDF template is uploaded)

Editable text area

Export as TXT, DOCX, or PDF

🔍 RAG (Retrieval Augmented Generation)

BioClinicalBERT embeddings

ChromaDB vector database

Retrieves similar patient cases to improve context

🗄️ Database Integration

MongoDB Atlas for real patient entries

ChromaDB persistent folder stored in repo

💬 AI Chat Assistant

Ask medical questions or direct the system to generate summaries.

🧠 Feedback Loop (Learning System)

You can “commit” a final edited summary, and the system adds it back into ChromaDB for future improvement.

🧱 Technology Stack
Component	Technology
Frontend	Streamlit
Vector Search	ChromaDB
Embeddings	BioClinicalBERT
LLM	Groq Cloud – Llama 3 (8B)
Patient Database	MongoDB Atlas
PDF / DOCX	ReportLab & python-docx
🌩️ Hybrid Cloud Architecture

Since Streamlit Cloud cannot run heavy models or Ollama, we use:

Local + Cloud Model Routing

If running locally → use Ollama Llama 3

If deployed → use Groq Cloud API

This makes the project fully deployable with zero GPU/RAM requirements.

📁 Project Structure
project/
│ app.py
│ config.py
│ requirements.txt
│ README.md
│
└── vector_db/
      └── chroma/
            (ChromaDB index files)

🔑 Environment Variables (Secrets)

In Streamlit Cloud → Advanced Settings → Secrets, add:

GROQ_API_KEY = "your_groq_key_here"
MONGO_URI = "your_mongo_atlas_uri_here"

▶️ Running Locally
1. Install dependencies
pip install -r requirements.txt

2. Start Ollama (if using locally)
ollama run llama3

3. Start the Streamlit App
streamlit run app.py

🌐 Deploying to Streamlit Cloud

Push your project to GitHub

Go to https://share.streamlit.io

Choose your repository

Set app.py as the entry file

Add secrets (Groq API, MongoDB URI)

Deploy

🧪 Testing

You may test with a sample patient:

Search patient by unit number

Use the chat assistant

Generate summaries

Export files

Commit the summary to the knowledge base (optional)

✨ Future Improvements

PDF-to-text patient extraction

HL7 integration

Multi-model inference

Role-based doctor login

Covering ICD-10 auto-coding

📜 License

MIT License — free for academic or clinical research use.