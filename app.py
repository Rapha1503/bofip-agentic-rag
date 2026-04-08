"""
BOFIP RAG Streamlit Application

A chat interface for querying French tax documentation.
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.hybrid import get_hybrid_retriever
from src.generation.llm import get_llm_client, get_current_model_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="BOFIP Assistant", page_icon="ðŸ“š", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    with st.spinner("Chargement de l'index de recherche..."):
        st.session_state.retriever = get_hybrid_retriever()

if "llm" not in st.session_state:
    st.session_state.llm = get_llm_client()


def format_sources(sources: list) -> str:
    """Format sources for display."""
    if not sources:
        return ""

    lines = ["\n\n---\n**Sources:**"]
    for s in sources:
        ref = s.get("boi_reference", "N/A")
        url = s.get("source_url", "")
        if url:
            lines.append(f"- [{ref}]({url})")
        else:
            lines.append(f"- {ref}")
    return "\n".join(lines)


def query_rag(question: str) -> dict:
    """
    Query the RAG system with SIMPLE retrieval.

    KISS approach: Just BM25 + Vector search, merge results, send to LLM.
    No HyDE, no complex post-processing, no LLM filtering.
    """
    # Simple retrieval: BM25 + Vector, merged by score
    chunks = st.session_state.retriever.search_simple(question, n_results=20)

    if not chunks:
        return {
            "answer": "Je n'ai pas trouve d'information pertinente dans la documentation BOFIP pour repondre a cette question.",
            "sources": [],
            "chunks_used": 0,
        }

    logger.info(f"Retrieved {len(chunks)} chunks with simple search")

    # Generate answer directly - no LLM filtering
    result = st.session_state.llm.generate_with_sources(
        question, chunks, use_cache=True
    )

    return result


# UI
st.title("ðŸ“š Assistant BOFIP")
st.caption(
    "Posez vos questions sur la fiscalite francaise - Reponses basees sur le BOFIP, le CGI et le LPF"
)

# Sidebar with info
with st.sidebar:
    st.header("A propos")
    st.markdown(
        """
    Cet assistant repond a vos questions fiscales en se basant
    sur la documentation officielle du **BOFIP**, du **CGI**
    et du **LPF**.

    **Fonctionnalites:**
    - Recherche hybride (semantique + mots-cles)
    - Citations des sources officielles
    - Liens vers les documents originaux

    ---

    **Exemples de questions:**
    - Quel est le taux de TVA pour la restauration?
    - Comment fonctionne le regime micro-BIC?
    - Quelles sont les conditions du credit impot recherche?
    """
    )

    st.divider()

    st.warning(
        """
    **Avertissement:** Cet outil fournit des informations
    a titre indicatif. Il ne remplace pas l'avis d'un
    expert-comptable ou d'un avocat fiscaliste.
    """
    )

    # Stats
    st.divider()
    st.subheader("Statistiques")
    st.metric(
        "Documents indexes", f"{st.session_state.retriever.vector_store.get_count():,}"
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez votre question sur la fiscalite..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            result = query_rag(prompt)

        # Format response with sources
        response = result["answer"]
        sources_text = format_sources(result.get("sources", []))
        full_response = response + sources_text

        st.markdown(full_response)

        # Show chunks used
        if result.get("chunks_used"):
            st.caption(f"*{result['chunks_used']} documents consultes*")

        # Show faithfulness guardrail outcome (compact audit signal)
        faith = result.get("faithfulness", {})
        if faith.get("enabled"):
            if faith.get("pass", False):
                st.caption("*Controle de fiabilite: OK*")
            else:
                st.warning(
                    f"Controle de fiabilite: abstention ({faith.get('reason', 'evidence insuffisante')})"
                )

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer - dynamic model name
st.divider()
st.caption(f"Propulse par {get_current_model_name()} via Groq | Donnees: BOFIP + CGI + LPF")

