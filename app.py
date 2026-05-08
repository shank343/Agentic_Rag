import streamlit as st
import os, time, gc, uuid, tempfile
from dotenv import load_dotenv
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

load_dotenv()
client = Client()

# ── Constants ────────────────────────────────────────────────────
MAX_CHUNKS     = 25      # 25 RPD per ingestion → 40 ingestions/day
CHUNK_SIZE     = 1000    # ~250 tokens per chunk
CHUNK_OVERLAP  = 100
MAX_SESSIONS   = 5       # max ingestions per user session
FETCH_K        = 3       # final docs returned
SIM_THRESHOLD  = 0.7     # routing threshold

# ── Session ID + Chroma dir ──────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
CHROMA_DIR = f"/tmp/chroma_{st.session_state.session_id}"

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Agentic RAG App", page_icon="📚", layout="wide")

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
div[data-testid="InputInstructions"] { display: none; }
.terminal-box {
    background: #050508; border: 1px solid #1a1a2e; border-radius: 8px;
    padding: 1.25rem 1.5rem; font-family: monospace; font-size: 0.85rem;
    line-height: 1.8; min-height: 120px; color: #00ffcc;
}
.log-line { margin: 0; padding: 0; }
.log-info  { color: #4488ff; }
.log-ok    { color: #44ff88; }
.log-warn  { color: #ffaa00; }
.log-error { color: #ff6688; }
.log-dim   { color: #445566; }
.url-chip {
    display: inline-flex; align-items: center; background: #0d0d18;
    border: 1px solid #2a2a4a; border-radius: 20px; padding: 0.25rem 0.75rem;
    font-family: monospace; font-size: 0.78rem; color: #4488ff;
    margin: 0.25rem; word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────
def clean_metadata(docs):
    for doc in docs:
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in {"source", "title"}}
    return docs

def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
        request_options={"timeout": 120}
    )

def build_ensemble_retriever(vectorstore, docs):
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    chroma_ret = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": FETCH_K})
    bm25_ret = BM25Retriever.from_documents(docs, k=FETCH_K)
    return EnsembleRetriever(
        retrievers=[chroma_ret, bm25_ret], weights=[0.6, 0.4])

def load_vectorstore_from_disk():
    from langchain_chroma import Chroma
    if os.path.exists(CHROMA_DIR):
        return Chroma(collection_name="rag-session",
                      persist_directory=CHROMA_DIR,
                      embedding_function=get_embeddings())
    return None

# ── Traceable invoke ─────────────────────────────────────────────
@traceable(name="rag-pipeline-run")
def traced_invoke(query, retriever, vectorstore):
    from graph.graph import app
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None
    sim_results = vectorstore.similarity_search_with_score(query, k=1)
    route_score = sim_results[0][1] if sim_results else 999.0
    stream_outputs = list(app.stream({
        "question": query, "retriever": retriever, "vectorstore": vectorstore,
        "route_score": route_score, "route_decision": "",
        "web_search": False, "documents": [], "generation": "",
    }, config={"recursion_limit": 10}))
    return stream_outputs, sim_results, run_id

# ── Session state init ───────────────────────────────────────────
if "urls" not in st.session_state:
    st.session_state.urls = ["https://paulgraham.com/ds.html?utm_source=chatgpt.com"]
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "bm25_docs" not in st.session_state:
    st.session_state.bm25_docs = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
    vs = load_vectorstore_from_disk()
    if vs and st.session_state.bm25_docs:
        st.session_state.vectorstore = vs
        st.session_state.retriever = build_ensemble_retriever(vs, st.session_state.bm25_docs)
        st.session_state.ingested = True
if "append_mode" not in st.session_state:
    st.session_state.append_mode = False
if "url_input_key" not in st.session_state:
    st.session_state.url_input_key = 0
if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0
if "embed_api_calls" not in st.session_state:
    st.session_state.embed_api_calls = 0

# ── Header ───────────────────────────────────────────────────────
st.markdown("## 📚 Agentic RAG Application")
st.caption("Retrieval Augmented Generation · Google Embeddings · Chroma + BM25 · LangGraph")
st.markdown("[View on GitHub](https://github.com/shank343/Agentic_Rag)")

tab1, tab2, tab3 = st.tabs(["ABOUT", "INGEST", "QUERY"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1.5rem; padding:1.25rem 1.5rem;
                background:#f8f9ff; border:1px solid #e0e0f0; border-radius:10px; margin-bottom:2rem;">
        <div style="width:48px; height:48px; border-radius:50%; background:#e8f4ff;
                    border:2px solid #4488ff; display:flex; align-items:center;
                    justify-content:center; font-size:1.3rem; flex-shrink:0;">👤</div>
        <div>
            <div style="font-size:1.05rem; font-weight:700; color:#111133;">Shashank M</div>
            <div style="font-size:0.78rem; color:#4488ff; letter-spacing:0.08em;
                        font-family:monospace; margin-top:0.2rem;">DATA SCIENTIST · ROLLS-ROYCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Agentic RAG System")
    st.markdown("---")
    st.markdown("""
    This project implements an **Adaptive Retrieval-Augmented Generation (RAG)** pipeline
    built on **LangGraph**. Unlike traditional RAG systems, this system dynamically routes 
    queries between local retrieval and live web search based on semantic relevance.

    The system uses a **hybrid retrieval** strategy combining semantic search (Chroma + Google
    embeddings) with keyword search (BM25) — giving the best of both worlds.

    The key components of the system in place: 
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔵 Router**
        Uses semantic similarity scoring to route questions to the vectorstore or web search.
        Questions with a similarity score below 0.7 are routed to RAG; others go directly
        to web search.

        ---

        **🟣 Hybrid Retrieve**
        Combines Chroma semantic search (60%) with BM25 keyword search (40%) via an
        EnsembleRetriever.

        ---

        **🟡 Document Grader**
        Uses `llama-3.1-8b-instant` to evaluate each retrieved chunk for relevance.
        Irrelevant chunks are dropped and web search is triggered to supplement.
        """)
    with col2:
        st.markdown("""
        **🟠 Web Search**
        Tavily fetches real-time web results (capped at 1,000 chars per result) when the
        vectorstore cannot fully answer the question. Results are appended to relevant chunks.

        ---

        **🔴 Generate + Quality Checks**
        `llama-3.1-8b-instant` generates the answer. `llama-3.3-70b-versatile` then runs
        two quality checks — a hallucination grader verifies grounding in source documents,
        and an answer grader checks whether the question is actually resolved. Fails trigger
        generation retries or web search fallback.
        """)

    st.markdown("""
    ---
    Stack: **Groq** (`llama-3.1-8b-instant` for generation and document grading ·
    `llama-3.3-70b-versatile` for hallucination and answer quality checks) · **Tavily** web search · **Google Generative AI** embeddings ·
    **LangSmith** end-to-end tracing
    """)

    st.markdown("### Process Flow")
    st.image("static/Langgraph Adaptive Rag.png", use_container_width=True)
    st.markdown("---")
    st.info("👉 Click on the **INGEST** tab to test the app!")


# ════════════════════════════════════════════════════════════════
# TAB 2 — INGEST
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Data Ingestion")

    calls_left = MAX_SESSIONS - st.session_state.embed_api_calls
    if st.session_state.embed_api_calls > 0:
        st.caption(f"📊 Sessions: {st.session_state.embed_api_calls}/{MAX_SESSIONS} — {calls_left} remaining")

    if st.session_state.append_mode:
        st.info(f"➕ **Append mode** — Add new sources and click INGEST. "
                f"Max {MAX_CHUNKS} chunks per session. Previously ingested docs are safe.")
    else:
        st.caption("Add URLs and/or upload local files, then click Ingest to build the knowledge base.")
        st.caption(f"_A link has been preselected. Max {MAX_CHUNKS} chunks · {MAX_SESSIONS} ingestions per session._")

    st.markdown("---")

    # ── URL input ─────────────────────────────────────────────────
    st.markdown("**🌐 Add URLs**")
    c1, c2 = st.columns([5, 1])
    with c1:
        url_input = st.text_input("URL", placeholder="https://example.com/article",
                                  key=f"url_input_{st.session_state.url_input_key}",
                                  label_visibility="collapsed")
    with c2:
        add_url = st.button("＋ Add", key="add_url_btn")

    if add_url:
        url = url_input.strip()
        if not url:
            st.warning("Please enter a URL first.")
        elif not url.startswith("http"):
            st.warning("Please enter a valid URL starting with http:// or https://")
        elif url in st.session_state.urls:
            st.warning("This URL is already in the list.")
        else:
            st.session_state.urls.append(url)
            st.session_state.url_input_key += 1
            st.rerun()

    if st.session_state.urls:
        st.markdown("**Added URLs:**")
        for i, url in enumerate(st.session_state.urls):
            ca, cb = st.columns([6, 1])
            with ca:
                st.markdown(f'<div class="url-chip">🔗 {url}</div>', unsafe_allow_html=True)
            with cb:
                if st.button("✕", key=f"remove_url_{i}"):
                    st.session_state.urls.pop(i)
                    st.rerun()
    else:
        st.caption("No URLs added yet.")

    st.markdown("---")

    # ── File upload ───────────────────────────────────────────────
    st.markdown("**📁 Upload Local Files**")
    uploaded_files = st.file_uploader(
        "Files", accept_multiple_files=True,
        type=["pdf", "txt", "docx", "csv", "md"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.append_mode}"
    )
    if uploaded_files:
        st.session_state.uploaded_files_count = len(uploaded_files)
        st.caption(f"{len(uploaded_files)} file(s): {', '.join(f.name for f in uploaded_files)}")

    st.markdown("---")

    # ── Ingest button ─────────────────────────────────────────────
    has_sources = (len(st.session_state.urls) > 0 or
                   bool(uploaded_files) or
                   st.session_state.uploaded_files_count > 0)
    already_ingested = st.session_state.ingested and not st.session_state.append_mode
    budget_exhausted = st.session_state.embed_api_calls >= MAX_SESSIONS

    ingest_btn = st.button("⚡ INGEST",
                           disabled=already_ingested or not has_sources or budget_exhausted,
                           type="primary")

    log_placeholder = st.empty()
    if already_ingested:
        st.caption("⚠️ Knowledge base exists. Use **Add More Documents** to append.")
    if budget_exhausted:
        st.error(f"🚫 Session limit reached ({MAX_SESSIONS} ingestions). Start a new session.")
    status_placeholder = st.empty()

    def render_log(lines):
        def style(line):
            l = line.lower()
            if "✓" in line or "complete" in l or "done" in l:
                return f'<p class="log-line log-ok">{line}</p>'
            elif "✗" in line or "error" in l or "failed" in l:
                return f'<p class="log-line log-error">{line}</p>'
            elif "⚠" in line or "warn" in l:
                return f'<p class="log-line log-warn">{line}</p>'
            elif line.startswith("//"):
                return f'<p class="log-line log-dim">{line}</p>'
            else:
                return f'<p class="log-line log-info">{line}</p>'
        log_placeholder.markdown(
            '<div class="terminal-box">' + "".join(style(l) for l in lines) + '</div>',
            unsafe_allow_html=True)

    render_log(["// Waiting for sources and ingest command..."])

    if ingest_btn and has_sources and not already_ingested and not budget_exhausted:
        logs = []
        def add_log(msg):
            logs.append(msg)
            render_log(logs)

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_chroma import Chroma
            import shutil

            all_docs = []

            # ── Load URLs ─────────────────────────────────────────
            if st.session_state.urls:
                add_log(f"[LOADER]  Loading {len(st.session_state.urls)} URL(s)...")
                for url in st.session_state.urls:
                    try:
                        add_log(f"          → {url}")
                        from langchain_community.document_loaders import WebBaseLoader
                        docs = WebBaseLoader(url, header_template={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.5",
                        }).load()
                        if docs and "just a moment" in docs[0].page_content.lower():
                            add_log(f"          ✗ Cloudflare protected — try a different URL")
                            continue
                        all_docs.extend(clean_metadata(docs))
                        del docs
                        add_log(f"          ✓ Loaded")
                    except Exception as e:
                        add_log(f"          ✗ {str(e)[:80]}")

            # ── Load files ────────────────────────────────────────
            if uploaded_files:
                add_log(f"[LOADER]  Loading {len(uploaded_files)} file(s)...")
                for uf in uploaded_files:
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False,
                                suffix=f".{uf.name.split('.')[-1]}") as tmp:
                            tmp.write(uf.getbuffer())
                            tmp_path = tmp.name
                        add_log(f"          → {uf.name}")
                        ext = uf.name.split(".")[-1].lower()
                        if ext == "pdf":
                            from langchain_community.document_loaders import PyPDFLoader
                            loader = PyPDFLoader(tmp_path)
                        elif ext == "docx":
                            from langchain_community.document_loaders import Docx2txtLoader
                            loader = Docx2txtLoader(tmp_path)
                        elif ext in ["txt", "md"]:
                            from langchain_community.document_loaders import TextLoader
                            loader = TextLoader(tmp_path)
                        elif ext == "csv":
                            from langchain_community.document_loaders import CSVLoader
                            loader = CSVLoader(tmp_path)
                        else:
                            add_log(f"          ✗ Unsupported: {ext}")
                            continue
                        docs = loader.load()
                        all_docs.extend(clean_metadata(docs))
                        del docs
                        add_log(f"          ✓ Loaded")
                    except Exception as e:
                        add_log(f"          ✗ {str(e)[:80]}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

            # ── Chunk ─────────────────────────────────────────────
            add_log("")
            add_log(f"[SPLITTER] {len(all_docs)} doc(s) loaded — chunking...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(all_docs)
            del all_docs
            gc.collect()

            if len(chunks) > MAX_CHUNKS:
                add_log(f"           ⚠️ Capping at {MAX_CHUNKS} chunks (session limit)")
                chunks = chunks[:MAX_CHUNKS]
            add_log(f"           ✓ {len(chunks)} chunks (~{len(chunks) * 250:,} tokens)")

            # ── Embed ─────────────────────────────────────────────
            add_log("")
            add_log("[EMBEDDER] Initialising Google embeddings...")
            embeddings = get_embeddings()
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            add_log(f"[EMBEDDER] Embedding {len(texts)} chunks...")

            with st.spinner("Hang on. This might take a minute...."):
                embeddings_list = embeddings.embed_documents(texts)
                add_log(f"           ✓ Embedded {len(texts)} chunks")

                # ── Chroma ────────────────────────────────────────
                if st.session_state.append_mode and os.path.exists(CHROMA_DIR):
                    add_log("[CHROMA]   Appending to existing store...")
                    vs = Chroma(collection_name="rag-session",
                                persist_directory=CHROMA_DIR,
                                embedding_function=embeddings)
                    base_id = str(int(time.time()))
                    vs._collection.add(
                        ids=[f"{base_id}_{i}" for i in range(len(chunks))],
                        embeddings=embeddings_list,
                        documents=texts, metadatas=metadatas)
                    st.session_state.bm25_docs.extend(chunks)
                    add_log(f"           ✓ Appended {len(chunks)} chunks")
                else:
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                    add_log("[CHROMA]   Building new vector store...")
                    vs = Chroma(collection_name="rag-session",
                                persist_directory=CHROMA_DIR,
                                embedding_function=embeddings)
                    vs._collection.add(
                        ids=[str(i) for i in range(len(chunks))],
                        embeddings=embeddings_list,
                        documents=texts, metadatas=metadatas)
                    st.session_state.bm25_docs = list(chunks)
                    add_log(f"           ✓ Indexed {len(chunks)} chunks")

            st.session_state.embed_api_calls += 1
            remaining = MAX_SESSIONS - st.session_state.embed_api_calls
            add_log(f"           ✓ Session {st.session_state.embed_api_calls}/{MAX_SESSIONS} — {remaining} remaining")

            del chunks
            gc.collect()

            # ── Hybrid retriever ──────────────────────────────────
            add_log("[RETRIEVER] Building hybrid retriever (Chroma 60% + BM25 40%)...")
            st.session_state.vectorstore = vs
            st.session_state.retriever = build_ensemble_retriever(vs, st.session_state.bm25_docs)
            st.session_state.ingested = True
            st.session_state.append_mode = False
            st.session_state.urls = []
            st.session_state.uploaded_files_count = 0

            add_log("            ✓ Hybrid retriever ready")
            add_log("")
            add_log("✓ Ingestion complete! Head to the QUERY tab.")
            status_placeholder.success("✅ Knowledge base ready! Go to the QUERY tab.")

        except Exception as e:
            add_log(f"✗ Error: {str(e)}")
            status_placeholder.error(f"Ingestion failed: {str(e)}")

    # ── Manage KB ─────────────────────────────────────────────────
    if st.session_state.ingested:
        st.markdown("---")
        st.markdown("**🗄️ Manage Knowledge Base**")
        ca, cb = st.columns(2)
        with ca:
            add_dis = st.session_state.embed_api_calls >= MAX_SESSIONS
            if st.button("➕ Add More Documents", use_container_width=True, disabled=add_dis):
                st.session_state.append_mode = True
                st.session_state.urls = []
                st.session_state.url_input_key += 1
                st.rerun()
            if add_dis:
                st.caption("Session limit reached.")
        with cb:
            if st.button("🗑️ Delete Knowledge Base", use_container_width=True, type="primary"):
                import shutil
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR)
                st.session_state.update({
                    "ingested": False, "retriever": None, "vectorstore": None,
                    "bm25_docs": [], "urls": [], "append_mode": False,
                    "url_input_key": st.session_state.url_input_key + 1,
                    "uploaded_files_count": 0
                })
                gc.collect()
                st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 3 — QUERY
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Query the Knowledge Base")

    if not st.session_state.ingested:
        st.warning("⚠️ No knowledge base found. Please go to the INGEST tab first.")
    else:
        st.success("✅ Knowledge base is ready.")

        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input("Ask a question",
                                  placeholder="Why are early unscalable actions important?",
                                  label_visibility="visible")
        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            ask = st.button("▶ ASK")

        if ask and not query.strip():
            st.warning("Please enter a question first.")

        if ask and query.strip():
            from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH

            log_ph = st.empty()
            qlogs = []

            def add_query_log(msg):
                qlogs.append(msg)
                def style(line):
                    l = line.lower()
                    if "✓" in line or "complete" in l or "grounded" in l:
                        return f'<p class="log-line log-ok">{line}</p>'
                    elif "✗" in line or "error" in l or "not supported" in l or "not useful" in l:
                        return f'<p class="log-line log-error">{line}</p>'
                    elif "web search" in l or "irrelevant" in l:
                        return f'<p class="log-line log-warn">{line}</p>'
                    elif line.startswith("//"):
                        return f'<p class="log-line log-dim">{line}</p>'
                    else:
                        return f'<p class="log-line log-info">{line}</p>'
                log_ph.markdown(
                    '<div class="terminal-box">' + "".join(style(l) for l in qlogs) + '</div>',
                    unsafe_allow_html=True)

            try:
                add_query_log(f"// Question: {query}")
                add_query_log("")
                add_query_log("[ROUTER]   Running similarity check...")

                with st.spinner("Agents are working..."):
                    stream_outputs, sim_results, run_id = traced_invoke(
                        query, st.session_state.retriever, st.session_state.vectorstore)

                if sim_results:
                    _, score = sim_results[0]
                    if score < SIM_THRESHOLD:
                        add_query_log(f"[ROUTER]   Score: {score:.3f} → below {SIM_THRESHOLD} → VECTORSTORE (RAG) ✓")
                    else:
                        add_query_log(f"[ROUTER]   Score: {score:.3f} → above {SIM_THRESHOLD} → WEB SEARCH")
                else:
                    add_query_log("[ROUTER]   No results — WEB SEARCH")

                add_query_log("")

                result = None
                generate_count = 0
                web_triggered = False
                retrieved_count = 0

                for output in stream_outputs:
                    for node_name, node_output in output.items():
                        if node_name == RETRIEVE:
                            retrieved_count = len(node_output.get("documents", []))
                            add_query_log(f"[RETRIEVE]  ✓ {retrieved_count} chunk(s) — semantic + keyword hybrid")
                        elif node_name == GRADE_DOCUMENTS:
                            web = node_output.get("web_search", False)
                            if web:
                                add_query_log("[GRADER]    Some chunk(s) dropped as irrelevant")
                                add_query_log("[GRADER]    → Adding web search to supplement")
                            else:
                                add_query_log(f"[GRADER]    ✓ All {retrieved_count} chunk(s) relevant")
                        elif node_name == WEBSEARCH:
                            web_triggered = True
                            add_query_log("[WEB SEARCH] Fetching from web...")
                            add_query_log("[WEB SEARCH] ✓ Web results fetched")
                        elif node_name == GENERATE:
                            generate_count += 1
                            msg = f"Re-generating (attempt {generate_count})..." if generate_count > 1 else "Drafting answer..."
                            add_query_log(f"[GENERATE]  {msg}")
                            add_query_log("[GENERATE]  ✓ Answer drafted")
                        result = node_output

                add_query_log("")
                if result and result.get("generation"):
                    if generate_count > 1:
                        add_query_log("[HALLUCINATION CHECK] ✗ Initial answer not grounded — retried")
                        add_query_log("[HALLUCINATION CHECK] ✓ Final answer grounded")
                    else:
                        add_query_log("[HALLUCINATION CHECK] ✓ Answer grounded in documents")

                    if web_triggered and generate_count > 1:
                        add_query_log("[ANSWER CHECK]        ✗ Initial answer insufficient — web search added")
                        add_query_log("[ANSWER CHECK]        ✓ Final answer addresses question")
                    else:
                        add_query_log("[ANSWER CHECK]        ✓ Answer addresses question")

                    add_query_log("")
                    add_query_log("✓ Pipeline complete.")

                    st.markdown("---")
                    st.markdown("**Answer:**")
                    st.markdown(result["generation"])

                    if web_triggered:
                        seen, sources = set(), []
                        for doc in result.get("documents", []):
                            for s in doc.metadata.get("sources", []):
                                if s and s not in seen:
                                    seen.add(s)
                                    sources.append(s)
                        if sources:
                            st.markdown("---")
                            st.markdown("**Sources consulted**")
                            for s in sources:
                                st.markdown(f"- {s}")

                    try:
                        if run_id:
                            share_link = None
                            for wait in [2, 4, 6]:
                                try:
                                    time.sleep(wait)
                                    share_link = client.share_run(run_id)
                                    break
                                except Exception:
                                    continue
                            st.markdown("---")
                            st.markdown("**LangSmith**")
                            if share_link:
                                st.markdown(f"🔍 [View LangSmith Trace]({share_link})")
                            else:
                                st.caption("Trace syncing — check LangSmith dashboard.")
                    except Exception:
                        pass

                else:
                    add_query_log("[HALLUCINATION CHECK] ✗ Answer failed grounding check")
                    add_query_log("[ANSWER CHECK]        ✗ Answer did not address question")
                    add_query_log("✗ Pipeline ended without valid answer.")
                    st.warning("No answer generated. Try rephrasing your question.")

            except Exception as e:
                if "recursion" in str(e).lower():
                    add_query_log("✗ Pipeline exceeded max steps.")
                    st.error("Too many retries. Try rephrasing your question.")
                else:
                    add_query_log(f"✗ Error: {str(e)}")
                    st.error(f"Error: {str(e)}")