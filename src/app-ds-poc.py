#!/usr/bin/env python3
import os
import jwt
import httpx
import openai
import weaviate
import streamlit as st
import base64
from pathlib import Path

from authlib.integrations.httpx_client import OAuth2Client
from pathlib import Path
from dotenv import load_dotenv
from weaviate import AuthApiKey, Client
from langchain_community.vectorstores import Weaviate
from embedding import get_embedding_function

# ─── LOAD ENV ─────────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "DS_POC")

# ─── CILOGON CONFIG ───────────────────────────────────────────
CILOGON_CLIENT_ID     = os.getenv("CILOGON_CLIENT_ID")
CILOGON_CLIENT_SECRET = os.getenv("CILOGON_CLIENT_SECRET")
CILOGON_REDIRECT_URI  = os.getenv("CILOGON_REDIRECT_URI")

AUTHORIZE_URL     = "https://cilogon.org/authorize"
TOKEN_URL         = "https://cilogon.org/oauth2/token"
REFRESH_TOKEN_URL = "https://cilogon.org/oauth2/token"
REVOKE_TOKEN_URL  = "https://cilogon.org/oauth2/revoke"
SCOPE             = "openid email profile org.cilogon.userinfo"

# Validate environment variables
if not all([OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY]):
    st.error("❌ Missing required environment variables. Please check your .env file.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# ─── STREAMLIT PAGE CONFIG ───────────────────────────────────
st.set_page_config(
    page_title="Dark Energy Survey - Sherpa",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CILOGON AUTH GATE ────────────────────────────────────────
def get_cilogon_client():
    return OAuth2Client(
        client_id=CILOGON_CLIENT_ID,
        client_secret=CILOGON_CLIENT_SECRET,
        redirect_uri=CILOGON_REDIRECT_URI,
        scope=SCOPE
    )

# Handle the callback — CILogon returns ?code=xxx in the URL
query_params = st.query_params

if "token" not in st.session_state:
    if "code" in query_params:
        if "oauth_code_used" not in st.session_state:
            st.session_state.oauth_code_used = True
            try:
                client_oauth = get_cilogon_client()
                token = client_oauth.fetch_token(
                    TOKEN_URL,
                    code=query_params["code"],
                    grant_type="authorization_code",
                    token_endpoint_auth_method="client_secret_post"
                )
                st.session_state.token = token
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                del st.session_state.oauth_code_used
                st.error(f"Authentication failed: {e}")
                st.stop()
        else:
            st.write("DEBUG: code already used, clearing params...")
            st.query_params.clear()
            st.rerun()
    else:
        # No token, no code — show login page
        client_oauth = get_cilogon_client()
        uri, state = client_oauth.create_authorization_url(AUTHORIZE_URL)
        st.session_state.oauth_state = state
        st.title("🔭 Data Sherpa")
        st.markdown("Please log in to access the Data Sherpa.")
        st.markdown(f"""
            <a href="{uri}" target="_self">
                <button style="
                    background-color:#005f9e;
                    color:white;
                    padding:12px 24px;
                    border:none;
                    border-radius:6px;
                    font-size:16px;
                    cursor:pointer;">
                    🔐 Login with CILogon
                </button>
            </a>
        """, unsafe_allow_html=True)
        st.stop()

# Decode user info from ID token
id_token = st.session_state.token.get("id_token")
user_info = jwt.decode(id_token, options={"verify_signature": False})
user_name  = user_info.get("name", "User")
user_email = user_info.get("email", "")

# ─── WEAVIATE + VECTORSTORE SETUP (once on import) ───────────
try:
    client = Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    
    # Check if schema exists
    if not client.schema.exists(CLASS_NAME):
        st.error("⚠️ Weaviate schema `{CLASS_NAME}` not initialized. Please run `python src/weaviate_setup.py` first.")
        st.stop()
    
    embedder = get_embedding_function()
    store = Weaviate(
        client=client,
        index_name=CLASS_NAME,
        text_key="content",
        embedding=embedder,
        by_text=False,      
        attributes=["source"],
    )
except Exception as e:
    st.error(f"❌ Failed to connect to Weaviate: {e}")
    st.info("Make sure you've run `python src/weaviate_setup.py` first.")
    st.stop()

# ─── EXPERT‑CONSULTANT PROMPT TEMPLATE ───────────────────────
SYSTEM_PROMPT = """You are an expert Data Concierge assistant for large astronomy collaborations, specifically the Dark Energy Survey (DES).

## Your Core Responsibility:
Help astronomers and researchers find relevant information in DES documentation, explain technical concepts, and clarify procedures.

## Your Approach:
- Answer ONLY using the provided context - do not use external knowledge
- Be concise but thorough - astronomers value precision
- Cite specific sources (PDF name and page) for each claim
- Use proper astronomical terminology and units
- If the context doesn't contain the answer, clearly state: "I don't find that information in the provided documents"
- Ask clarifying questions when requests are ambiguous

## Your Limitations:
- You cannot access live databases or execute queries
- You only know what's in the provided context
- For information not in your context, recommend consulting DES help desks

## Tone:
Professional yet approachable - a knowledgeable colleague helping researchers work efficiently.
"""

# ─── RAG FUNCTION ─────────────────────────────────────────────
def generate_response(question: str, k: int = 5, show_scores: bool = False) -> str:
    try:
        # Retrieve top‑k chunks
        docs_and_scores = store.similarity_search_with_score(question, k=k)
        
        if not docs_and_scores:
            return "I couldn't find any relevant information in the documents. Please try rephrasing your question."
        
        # Build the "context" block
        context = "\n\n---\n\n".join(doc.page_content for doc, _ in docs_and_scores)
        
        # Compose OpenAI Chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Context:
{context}

---

Question: {question}

Please answer using ONLY the context above. Cite each fact with the source PDF filename and page number if available. If the answer is not in the context, say so clearly."""}
        ]
        
        # Call OpenAI
        resp = openai.chat.completions.create(
            model="gpt-4o",  # Fixed model name
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content.strip()
        
        # List unique sources with better formatting
        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in docs_and_scores})
        sources_md = "\n".join(f"- `{s}`" for s in sources)
        
        # Build response with sources
        response = f"{answer}\n\n**Sources:**\n{sources_md}"
        
        # Optionally add relevance scores for debugging
        if show_scores:
            scores_info = "\n".join(
                f"- `{doc.metadata.get('source', 'unknown')}`: {score:.3f}" 
                for doc, score in docs_and_scores
            )
            response += f"\n\n**Relevance Scores (Debug):**\n{scores_info}"
        
        return response
            
    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}\nPlease try again or rephrase your question."

# ─── SIDEBAR ──────────────────────────────────────────────────
def img_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    # Change mime if not PNG
    return f"data:image/png;base64,{b64}"

with st.sidebar:
    col_left, col_mid, col_right = st.columns([1, 3, 1])
    with col_mid:
        uri = img_data_uri("src/des_logo.png")
        st.markdown(
            f"""
            <a href="https://www.darkenergysurvey.org" target="_blank" rel="noopener noreferrer"
               style="display:flex; justify-content:center;">
              <img src="{uri}" style="width:256px; height:auto; display:block; cursor:pointer;" alt="DES Logo"/>
            </a>
            """,
            unsafe_allow_html=True,
        )
    
    st.title("DES Sherpa")
    st.markdown("""
        This Sherpa provides information based on all the scientific papers produced by the 
        [Dark Energy Survey](https://www.darkenergysurvey.org/) (DES) project.
        
        **How to use:**
        Ask detailed questions about DES technical documentation and receive precise, 
        cited answers—powered by OpenAI GPT-4 and Weaviate.
        
        ⚠️ **Note:** AI generated answers may contain inaccuracies. Always verify 
        critical information with official DES documentation.
    """)

    st.divider()

    # ─── USER INFO & LOGOUT ───────────────────────────────────
    st.markdown(f"👤 **{user_name}**")
    st.markdown(f"`{user_email}`")
    if st.button("Logout"):
        del st.session_state.token
        st.rerun()

    st.divider()

    # Debug options
    st.subheader("Debug Options")
    show_relevance_scores = st.checkbox(
        "Show relevance scores",
        value=False,
        help="Display similarity scores for retrieved documents (useful for debugging)"
    )

# ─── STREAMLIT CHAT UI ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """👋 Welcome to the DES Sherpa!  
Ask me anything about the Dark Energy Survey project.

**Example questions:**
- When did the DES project start?
- How much did the DES project cost?
- What have been the main discoveries of the DES project?
- What instruments does DES use?
"""
    }]

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_q := st.chat_input("Type your question here…"):
    # Append user message
    st.session_state.messages.append({"role":"user","content":user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # Generate & display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_response(user_q, k=5, show_scores=show_relevance_scores)
            st.markdown(reply)

    # Save assistant message
    st.session_state.messages.append({"role":"assistant","content":reply})
