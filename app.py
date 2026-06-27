"""
Quran QA Application
====================
A fully local, offline-capable semantic search engine for the Quran.
Uses: Streamlit (UI) + ChromaDB (vector store) + SentenceTransformers (embeddings)
Zero paid APIs. Zero internet required after first model download.
"""

import os
import re
import time
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Quran QA",
    page_icon="☪️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — dark teal / gold Islamic palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d1f1a;
    --surface:   #122920;
    --border:    #1e4035;
    --gold:      #c9a84c;
    --gold-dim:  #8a6d2f;
    --text:      #e8dfc8;
    --muted:     #7a9e8e;
    --accent:    #2a6b52;
}

/* ── Global background ── */
.stApp { background-color: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }

/* ── Typography ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Amiri', serif; color: var(--gold); }

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.masthead .arabic {
    font-family: 'Amiri', serif;
    font-size: 2.4rem;
    color: var(--gold);
    line-height: 1.4;
    margin-bottom: 0.3rem;
    direction: rtl;
}
.masthead .subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: var(--accent);
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    margin: 0.6rem 0 0.6rem 3rem;
    color: var(--text);
    font-size: 0.95rem;
    border: 1px solid var(--border);
}
.bubble-bot {
    background: var(--surface);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 3rem 0.6rem 0;
    border: 1px solid var(--border);
    font-size: 0.93rem;
    line-height: 1.7;
}

/* ── Result cards ── */
.result-card {
    background: rgba(201,168,76,0.06);
    border: 1px solid var(--gold-dim);
    border-left: 4px solid var(--gold);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-family: 'Amiri', serif;
    font-size: 1.05rem;
    line-height: 1.9;
}
.result-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: var(--gold);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.score-badge {
    display: inline-block;
    background: var(--gold-dim);
    color: var(--bg);
    font-size: 0.68rem;
    padding: 1px 7px;
    border-radius: 10px;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ── Not-found banner ── */
.not-found {
    background: rgba(180,50,50,0.12);
    border: 1px solid #7a2f2f;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    color: #e8a0a0;
    font-size: 0.93rem;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: var(--gold) !important;
    color: var(--bg) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stFileUploader { border: 1px dashed var(--gold-dim) !important; border-radius: 8px !important; }
.stSpinner > div { color: var(--gold) !important; }
div[data-testid="stNumberInput"] input { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAZY IMPORTS  (heavy libs loaded only once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load the local SentenceTransformer model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# TEXT PARSING
# ─────────────────────────────────────────────
# Patterns that match common Quran translation formats, e.g.:
#   1:1  OR  (1:1)  OR  [1:1]  OR  1. Al-Fatihah 1  etc.
_VERSE_RE = re.compile(
    r"""
    (?:
        \[?                        # optional [
        (\d{1,3})                  # surah number
        [:\.\-]                    # separator
        (\d{1,3})                  # ayah number
        \]?                        # optional ]
    )
    """,
    re.VERBOSE,
)

# Surah name lookup (1–114) for richer citations
SURAH_NAMES = {
    1:"Al-Fatihah",2:"Al-Baqarah",3:"Ali 'Imran",4:"An-Nisa",5:"Al-Ma'idah",
    6:"Al-An'am",7:"Al-A'raf",8:"Al-Anfal",9:"At-Tawbah",10:"Yunus",
    11:"Hud",12:"Yusuf",13:"Ar-Ra'd",14:"Ibrahim",15:"Al-Hijr",
    16:"An-Nahl",17:"Al-Isra",18:"Al-Kahf",19:"Maryam",20:"Ta-Ha",
    21:"Al-Anbiya",22:"Al-Hajj",23:"Al-Mu'minun",24:"An-Nur",25:"Al-Furqan",
    26:"Ash-Shu'ara",27:"An-Naml",28:"Al-Qasas",29:"Al-'Ankabut",30:"Ar-Rum",
    31:"Luqman",32:"As-Sajdah",33:"Al-Ahzab",34:"Saba",35:"Fatir",
    36:"Ya-Sin",37:"As-Saffat",38:"Sad",39:"Az-Zumar",40:"Ghafir",
    41:"Fussilat",42:"Ash-Shura",43:"Az-Zukhruf",44:"Ad-Dukhan",45:"Al-Jathiyah",
    46:"Al-Ahqaf",47:"Muhammad",48:"Al-Fath",49:"Al-Hujurat",50:"Qaf",
    51:"Adh-Dhariyat",52:"At-Tur",53:"An-Najm",54:"Al-Qamar",55:"Ar-Rahman",
    56:"Al-Waqi'ah",57:"Al-Hadid",58:"Al-Mujadila",59:"Al-Hashr",60:"Al-Mumtahanah",
    61:"As-Saf",62:"Al-Jumu'ah",63:"Al-Munafiqun",64:"At-Taghabun",65:"At-Talaq",
    66:"At-Tahrim",67:"Al-Mulk",68:"Al-Qalam",69:"Al-Haqqah",70:"Al-Ma'arij",
    71:"Nuh",72:"Al-Jinn",73:"Al-Muzzammil",74:"Al-Muddaththir",75:"Al-Qiyamah",
    76:"Al-Insan",77:"Al-Mursalat",78:"An-Naba",79:"An-Nazi'at",80:"'Abasa",
    81:"At-Takwir",82:"Al-Infitar",83:"Al-Mutaffifin",84:"Al-Inshiqaq",85:"Al-Buruj",
    86:"At-Tariq",87:"Al-A'la",88:"Al-Ghashiyah",89:"Al-Fajr",90:"Al-Balad",
    91:"Ash-Shams",92:"Al-Layl",93:"Ad-Duhaa",94:"Ash-Sharh",95:"At-Tin",
    96:"Al-'Alaq",97:"Al-Qadr",98:"Al-Bayyinah",99:"Az-Zalzalah",100:"Al-'Adiyat",
    101:"Al-Qari'ah",102:"At-Takathur",103:"Al-'Asr",104:"Al-Humazah",105:"Al-Fil",
    106:"Quraysh",107:"Al-Ma'un",108:"Al-Kawthar",109:"Al-Kafirun",110:"An-Nasr",
    111:"Al-Masad",112:"Al-Ikhlas",113:"Al-Falaq",114:"An-Nas",
}

def format_citation(surah_num: int, ayah_num: int) -> str:
    name = SURAH_NAMES.get(surah_num, f"Surah {surah_num}")
    return f"Surah {surah_num} ({name}), Ayah {ayah_num}"


def parse_quran_text(raw_text: str) -> list[dict]:
    """
    Split raw Quran text into verse-level chunks.
    Each chunk: {"text": str, "citation": str, "surah": int, "ayah": int}

    Strategy:
      1. Try to find verse markers (S:V pattern).
      2. Fall back to paragraph chunking if no markers found.
    """
    chunks = []

    # ── Strategy 1: verse-by-verse splitting ──────────────────────────
    lines = raw_text.splitlines()
    current_lines = []
    current_surah = 0
    current_ayah = 0
    found_markers = False

    for line in lines:
        match = _VERSE_RE.search(line)
        if match:
            found_markers = True
            # Save previous chunk
            if current_lines and current_surah:
                text_block = " ".join(current_lines).strip()
                if text_block:
                    chunks.append({
                        "text": text_block,
                        "citation": format_citation(current_surah, current_ayah),
                        "surah": current_surah,
                        "ayah": current_ayah,
                    })
            current_surah = int(match.group(1))
            current_ayah = int(match.group(2))
            # Keep the rest of the line after the marker as verse text
            after_marker = line[match.end():].strip()
            current_lines = [after_marker] if after_marker else []
        else:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    # Flush last chunk
    if current_lines and current_surah:
        text_block = " ".join(current_lines).strip()
        if text_block:
            chunks.append({
                "text": text_block,
                "citation": format_citation(current_surah, current_ayah),
                "surah": current_surah,
                "ayah": current_ayah,
            })

    # ── Strategy 2: paragraph fallback ───────────────────────────────
    if not found_markers or len(chunks) < 10:
        chunks = []
        paragraphs = re.split(r"\n{2,}", raw_text)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) > 30:  # skip very short fragments
                chunks.append({
                    "text": para,
                    "citation": f"Block {i + 1}",
                    "surah": 0,
                    "ayah": i + 1,
                })

    return chunks


# ─────────────────────────────────────────────
# CHROMA VECTOR STORE
# ─────────────────────────────────────────────
CHROMA_DIR = "./quran_chroma_db"

@st.cache_resource(show_spinner=False)
def build_vector_store(_chunks: tuple, _embedder):
    """
    Build (or reload) a ChromaDB collection from parsed chunks.
    Accepts a tuple of dicts so Streamlit can hash it.
    """
    import chromadb
    chunks = list(_chunks)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh build every time the app is run with a new file
    try:
        client.delete_collection("quran")
    except Exception:
        pass

    collection = client.create_collection(
        name="quran",
        metadata={"hnsw:space": "cosine"},
    )

    texts     = [c["text"]     for c in chunks]
    citations = [c["citation"] for c in chunks]
    ids       = [f"v{i}"       for i in range(len(chunks))]

    # Embed in batches to avoid memory spikes
    BATCH = 256
    all_embeddings = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        embs  = _embedder.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embs)

    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        metadatas=[{"citation": c} for c in citations],
        ids=ids,
    )

    return collection


def semantic_search(query: str, collection, embedder, top_k: int = 5) -> list[dict]:
    """
    Embed the query and retrieve top_k most similar verse chunks.
    Returns list of {"text", "citation", "score"}.
    """
    query_emb = embedder.encode([query]).tolist()
    results   = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        similarity = round(1 - dist, 3)   # cosine distance → similarity
        hits.append({
            "text":     doc,
            "citation": meta["citation"],
            "score":    similarity,
        })
    return hits


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "collection"    not in st.session_state: st.session_state.collection    = None
if "indexed"       not in st.session_state: st.session_state.indexed       = False
if "file_name"     not in st.session_state: st.session_state.file_name     = ""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☪️ Quran QA Setup")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Quran text (.txt or .pdf)",
        type=["txt", "pdf"],
        help="Use a plain-text or PDF translation file.",
    )

    top_k = st.number_input(
        "Results to retrieve", min_value=1, max_value=10, value=4, step=1
    )

    score_threshold = st.slider(
        "Minimum relevance score", min_value=0.0, max_value=1.0, value=0.25, step=0.01,
        help="Results below this similarity score are discarded.",
    )

    if uploaded and (uploaded.name != st.session_state.file_name):
        with st.spinner("📖 Parsing & indexing — please wait…"):
            # Read raw text
            if uploaded.name.endswith(".pdf"):
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded) as pdf:
                        raw = "\n".join(
                            page.extract_text() or "" for page in pdf.pages
                        )
                except ImportError:
                    st.error("Install pdfplumber: `pip install pdfplumber`")
                    st.stop()
            else:
                raw = uploaded.read().decode("utf-8", errors="replace")

            chunks  = parse_quran_text(raw)
            embedder = load_embedder()
            coll     = build_vector_store(tuple(chunks), embedder)

            st.session_state.collection = coll
            st.session_state.indexed    = True
            st.session_state.file_name  = uploaded.name
            st.session_state.chat_history = []

        st.success(f"✅ Indexed {len(chunks)} verse chunks")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<p style='color:#7a9e8e;font-size:0.75rem;margin-top:1rem'>"
        "100% local · No API keys · No internet required after setup</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="arabic">بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</div>
    <div class="subtitle">Quran · Semantic Question-Answering · Exact Verse Retrieval</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NOT-YET-INDEXED STATE
# ─────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("⬅️  Upload your Quran text file in the sidebar to begin.", icon="📂")
    st.stop()

# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────
with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask a question about the Quran",
            placeholder="e.g. What does the Quran say about patience?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Ask")

# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────
if submitted and query.strip():
    user_q = query.strip()
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.markdown(f'<div class="bubble-user">🧑 {user_q}</div>', unsafe_allow_html=True)

    with st.spinner("Searching the Quran…"):
        embedder = load_embedder()
        hits     = semantic_search(
            user_q,
            st.session_state.collection,
            embedder,
            top_k=int(top_k),
        )

    # Filter by score threshold
    relevant = [h for h in hits if h["score"] >= score_threshold]

    if not relevant:
        answer_html = (
            '<div class="not-found">'
            '⚠️ <strong>I cannot find the answer to this in the provided text.</strong>'
            '</div>'
        )
    else:
        cards = ""
        for hit in relevant:
            score_pct = int(hit["score"] * 100)
            cards += (
                f'<div class="result-card">'
                f'  <div class="result-meta">'
                f'    📖 {hit["citation"]}'
                f'    <span class="score-badge">{score_pct}% match</span>'
                f'  </div>'
                f'  {hit["text"]}'
                f'</div>'
            )
        answer_html = cards

    st.markdown(f'<div class="bubble-bot">{answer_html}</div>', unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": answer_html})
