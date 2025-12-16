# 📚 Data Sherpa Proof of Concept
## RAG Chatbot System with OpenAI & Weaviate

A Retrieval-Augmented Generation (RAG) system built as a Proof Of Concept (POC) for the CAPS' Data Concierge Project. This version features **a specialized chatbot** that semantically queries documentation from the [Dark Energy Survey](https://www.darkenergysurvey.org/) (DES) knowledge base. The CAPS is the [Center for AstroPhysical Surveys](https://caps.ncsa.illinois.edu/) at the [University of Illinois](https://www.illinois.edu)' [National Center for Supercomputing Applications](https://ncsa.illinois.edu/).

Powered by **OpenAI GPT-4.1**, **Weaviate** vector database, and a clean **Streamlit UI**.

---

## 🚀 Features

- **Specialized Chatbot** with a specific test set of DES documents.
- **Embeddings** and **responses** generated via OpenAI API
- **Vector search** using Weaviate with semantic retrieval
- **Interactive chat UI** built with Streamlit
- Fully dockerizable & cloud‑ready deployment

---

## 🧰 Prerequisites

- Python 3.10+
- A valid OpenAI API key
- Weaviate running locally or remotely
- Git

---

## ⚙️ Setup

1. **Clone the repository**

```bash
git clone git@github.com:lacdacon/dc-poc.git
cd dc-poc
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file with the below information. This assumes you have set up previsouly a Weaviate instance where you can connect and create your vector data base.

```bash
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_bd_url
WEAVIATE_API_KEY=optional_if_required_from_your_weaviate_db
```

---

## 📥 Ingest Reports

1. We have provided the documents to test out this ChatBot; but if you want to test it with your own data, place your own PDF documents in the following folder:
   - `src/data/documents`

2. Run the document loader script:

```bash
python src/weaviate_setup.py
```

Use `--reset` to clear and rebuild the index.

---

## 💬 Start the Streamlit App

```bash
streamlit run src/app-dc-poc.py --server.address 0.0.0.0 --server.port 8501
```

Then open your browser at `http://localhost:8501` (or your VM IP).

You’ll be able to:
- See the ChatBot display on screen.
- Enter natural language queries.
- See answers with **source citations**.
---


## 📫 Contact

For access issues or bugs, please email **fabs@illinois.edu**
