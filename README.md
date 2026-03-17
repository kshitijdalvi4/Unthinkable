# Unthinkable - JSO (Job Search Optimizer)
*Last Updated: March 17, 2026*
 (Phase-2)

Welcome to the **Job Search Optimiser (JSO)** project. This is an agentic AI ecosystem designed to streamline career growth through intelligent job discovery, automated application handling, and resume-centric RAG intelligence.

---

## 🌟 Key Features

- **Autonomous Job Discovery**: A multi-agent system that navigates LinkedIn and external career sites to find the best-matched roles.
- **Resume Chat (RAG)**: A LangChain and ChromaDB-powered conversational interface for querying your professional history and skills.
- **Google OAuth Persistence**: Seamless, secure authentication with persistent user profiles and resume data.
- **Synchronized Ingestion**: Intelligent resume pipelines that automatically update vector embeddings whenever a new resume is uploaded.
- **Smart Form Filling**: Context-aware automation that detects and interacts with "Easy Apply" flows and external job forms.

---

## 🛠️ Technical Stack

- **Frontend**: React (Vite), TailwindCSS, Framer Motion, Lucide icons.
- **Backend**: FastAPI (Python), LangGraph, LangChain.
- **Database**: MongoDB (Identity & History), ChromaDB (Vector store for RAG).
- **AI/ML**: Google Gemini 2.5 Flash, Google Gemini Embeddings (`text-embedding-004`).
- **Automation**: Playwright (Browser Agent).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js (for frontend)
- Google Gemini API Key

### Setup
1. **Backend**:
   - `cd backend`
   - `pip install -r requirements.txt`
   - Set up `.env` with your `GEMINI_API_KEY` and `MONGO_URI`.
   - `python main.py`
2. **Frontend**:
   - `cd frontend`
   - `npm install`
   - `npm run dev`

---

## 📂 Project Documentation (For Recruiters)

I have prepared formal documents for the Phase-2 assignment review:
- **Assignment Proposal**: [View Proposal](./brain/JSO_Phase2_Assignment.md)
- **Submission & Deployment Guide**: [View Guide](./brain/RECRUITER_SUBMISSION_GUIDE.md)
- **Feature Walkthrough**: [View Walkthrough](./brain/walkthrough.md)

---
Developed by **Group JSO-AI-01** as part of the **Job Search Optimiser Phase-2: Agentic Career Intelligence Development**.
