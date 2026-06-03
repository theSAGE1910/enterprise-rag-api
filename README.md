# Enterprise RAG API 🏢

A production-ready Retrieval-Augmented Generation (RAG) backend API built with FastAPI. This system securely grounds AI responses in internal corporate documents using semantic vector search.

## 🚀 Architecture Overview
This project transforms static text documents into high-dimensional semantic vectors, stores them in a highly optimized PostgreSQL database, and exposes an asynchronous REST API. When a user asks a question, the API performs a mathematical similarity search to find the most relevant business context and uses a Large Language Model to synthesize a precise, accurate answer.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python, Asynchronous)
* **AI Core:** LangChain, Google Gemini (`gemini-2.5-flash`)
* **Embeddings:** Google GenAI (`gemini-embedding-001`, 3072 dimensions)
* **Database:** PostgreSQL with `pgvector` (via Docker)
* **ORM:** SQLAlchemy 2.0 (Async)

## 📋 Prerequisites
Before running this project locally, ensure you have the following installed:
* [Python 3.10+](https://www.python.org/downloads/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* A [Google AI Studio API Key](https://aistudio.google.com/)

## ⚙️ Local Setup & Installation

**1. Clone the repository and set up the environment**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
