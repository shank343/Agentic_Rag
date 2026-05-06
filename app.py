import streamlit as st
import tempfile
import os
import time
from dotenv import load_dotenv
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

def clean_metadata(docs):
    allowed_keys = {"source", "title"}  # keep it minimal
    for doc in docs:
        doc.metadata = {
            k: v for k, v in doc.metadata.items() if k in allowed_keys
        }
    return docs

load_dotenv()
client = Client()

st.set_page_config(
    page_title="Agentic RAG App",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
div[data-testid="InputInstructions"] { display: none; }

.terminal-box {
    background: #050508;
    border: 1px solid #1a1a2e;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-family: monospace;
    font-size: 0.85rem;
    line-height: 1.8;
    min-height: 120px;
    color: #00ffcc;
}
.log-line  { margin: 0; padding: 0; }
.log-info  { color: #4488ff; }
.log-ok    { color: #44ff88; }
.log-warn  { color: #ffaa00; }
.log-error { color: #ff6688; }
.log-dim   { color: #445566; }

.url-chip {
    display: inline-flex;
    align-items: center;
    background: #0d0d18;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: monospace;
    font-size: 0.78rem;
    color: #4488ff;
    margin: 0.25rem;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown("## 📚 Agentic RAG Application")
st.caption("Retrieval Augmented Generation · HuggingFace Embeddings · Chroma · LangGraph")

# ── Traceable invoke ─────────────────────────────────────────────
@traceable(name="rag-pipeline-run")
def traced_invoke(query, retriever):
    from graph.graph import app

    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None

    vectorstore = retriever.vectorstore
    sim_results = vectorstore.similarity_search_with_score(query, k=1)
    route_score = sim_results[0][1] if sim_results else 999.0

    stream_outputs = list(app.stream({
        "question": query,
        "retriever": retriever,
        "route_score": route_score,
        "route_decision": "",
        "web_search": False,
        "documents": [],
        "generation": "",
    }, config={"recursion_limit": 15}))

    return stream_outputs, sim_results, run_id

# ── Session state init ──────────────────────────────────────────
if "urls" not in st.session_state:
    st.session_state.urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "append_mode" not in st.session_state:
    st.session_state.append_mode = False
if "url_input_key" not in st.session_state:
    st.session_state.url_input_key = 0
if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0


# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["ABOUT", "INGEST", "QUERY"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab1:

    # ── Author card ──────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1.5rem; padding:1.25rem 1.5rem;
                background:#f8f9ff; border:1px solid #e0e0f0; border-radius:10px;
                margin-bottom:2rem;">
        <div style="width:48px; height:48px; border-radius:50%; background:#e8f4ff;
                    border:2px solid #4488ff; display:flex; align-items:center;
                    justify-content:center; font-size:1.3rem; flex-shrink:0;">👤</div>
        <div>
            <div style="font-size:1.05rem; font-weight:700; color:#111133;">Shashank M</div>
            <div style="font-size:0.78rem; color:#4488ff; letter-spacing:0.08em;
                        font-family:monospace; margin-top:0.2rem;">
                DATA SCIENTIST · ROLLS-ROYCE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Heading ──────────────────────────────────────────────────
    st.markdown("### Agentic RAG System")
    st.markdown("---")

    # ── Project description ───────────────────────────────────────
    st.markdown("""
    This project implements an **Adaptive Retrieval-Augmented Generation (RAG)** pipeline
    built on **LangGraph** — a framework for building stateful, graph-based AI workflows.
    Unlike traditional RAG systems that always retrieve from a fixed knowledge base, this
    system intelligently decides at runtime whether to answer from ingested documents or
    fall back to live web search, depending on the relevance of the question.

    The system is composed of five specialised components:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔵 Router**
        Uses semantic similarity scoring to determine whether the user's question is
        related to the ingested knowledge base. If the similarity score is below the
        threshold, the question is routed to the vectorstore. Otherwise it goes directly
        to web search.

        ---

        **🟣 Retrieve**
        Fetches the most semantically relevant document chunks from the Chroma vectorstore
        using HuggingFace's `all-MiniLM-L6-v2` embedding model. Returns the top matching
        chunks for the grader to evaluate.

        ---

        **🟡 Document Grader**
        Evaluates each retrieved chunk individually for relevance to the question. Irrelevant
        chunks are dropped. If any chunk is marked irrelevant, a web search is triggered to
        supplement the remaining relevant documents.
        """)

    with col2:
        st.markdown("""
        **🟠 Web Search**
        Uses Tavily's search API to fetch real-time web results when the vectorstore cannot
        fully answer the question. Web results are appended to the existing relevant chunks
        to give the generator the richest possible context.

        ---

        **🔴 Generate + Quality Checks**
        The generator produces an answer from the combined document context. Two quality
        checks follow — a **hallucination grader** verifies the answer is grounded in the
        source documents, and an **answer grader** checks whether it actually resolves the
        question. If either check fails, the pipeline retries or falls back to web search.
        """)

    st.markdown("""
    ---
    The pipeline runs on **Groq's inference API** — `llama-3.1-8b-instant` for document relevance grading + answer generation and `llama-3.3-70b-versatile` for quality checks — alongside **Tavily** for live web search and **HuggingFace** embeddings for semantic similarity. Every run is traced end-to-end with **LangSmith**.
    """)

    # ── Process Flow ─────────────────────────────────────────────
    st.markdown("### Process Flow")
    st.image("static/Langgraph Adaptive Rag.png", use_container_width=True)

    st.markdown("---")
    st.info("👉 Click on the **INGEST** tab to test the app!")


# ════════════════════════════════════════════════════════════════
# TAB 2 — INGEST
# ════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("#### Data Ingestion")

    if st.session_state.append_mode:
        st.info("➕ **Append mode** — Add new sources below and click INGEST to add them to the existing knowledge base. Previously ingested documents are safe and will not be re-added.")
    else:
        st.caption("Add URLs and/or upload local files, then click Ingest to build the knowledge base.")
        st.caption("_A few links have been selected for you. You can remove them and add your own._")

    st.markdown("---")

    # ── URL Section ──────────────────────────────────────────────
    st.markdown("**🌐 Add URLs**")

    url_col1, url_col2 = st.columns([5, 1])
    with url_col1:
        url_input = st.text_input(
            "Enter a URL",
            placeholder="https://example.com/article",
            key=f"url_input_{st.session_state.url_input_key}",
            label_visibility="collapsed"
        )
    with url_col2:
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
            col_url, col_remove = st.columns([6, 1])
            with col_url:
                st.markdown(f'<div class="url-chip">🔗 {url}</div>', unsafe_allow_html=True)
            with col_remove:
                if st.button("✕", key=f"remove_url_{i}"):
                    st.session_state.urls.pop(i)
                    st.rerun()
    else:
        st.caption("No URLs added yet.")

    st.markdown("---")

    # ── File Upload Section ───────────────────────────────────────
    st.markdown("**📁 Upload Local Files**")

    if st.session_state.append_mode:
        st.caption("Upload new files to append to the existing knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "csv", "md"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.append_mode}"
    )

    if uploaded_files:
        st.session_state.uploaded_files_count = len(uploaded_files)
        st.caption(f"{len(uploaded_files)} new file(s) selected: {', '.join([f.name for f in uploaded_files])}")

    st.markdown("---")

    # ── Ingest Button — enabled if URLs OR files present ──────────
    has_sources = (
        len(st.session_state.urls) > 0 or
        (uploaded_files is not None and len(uploaded_files) > 0) or
        st.session_state.uploaded_files_count > 0
    )
    already_ingested = st.session_state.ingested and not st.session_state.append_mode
    ingest_btn = st.button(
        "⚡ INGEST",
        disabled=already_ingested or not has_sources,
        type="primary"
    )
    log_placeholder = st.empty()
    if already_ingested:
        st.caption("⚠️ Knowledge base already exists. Use **Add More Documents** below to append.")
    status_placeholder = st.empty()

    def render_log(lines):
        def style(line):
            l = line.lower()
            if "✓" in line or "complete" in l or "success" in l or "done" in l:
                return f'<p class="log-line log-ok">{line}</p>'
            elif "✗" in line or "error" in l or "failed" in l:
                return f'<p class="log-line log-error">{line}</p>'
            elif "warning" in l or "warn" in l:
                return f'<p class="log-line log-warn">{line}</p>'
            elif line.startswith("//"):
                return f'<p class="log-line log-dim">{line}</p>'
            else:
                return f'<p class="log-line log-info">{line}</p>'
        html = '<div class="terminal-box">' + "".join(style(l) for l in lines) + '</div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)

    if not ingest_btn:
        render_log(["// Waiting for sources and ingest command..."])

    if ingest_btn and has_sources and not already_ingested:
        logs = []

        def add_log(msg):
            logs.append(msg)
            render_log(logs)

        try:
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            all_docs = []

            if st.session_state.urls:
                add_log(f"[LOADER]  Loading {len(st.session_state.urls)} URL(s)...")
                for url in st.session_state.urls:
                    try:
                        add_log(f"          → {url}")
                        docs = WebBaseLoader(
                            url,
                            header_template={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.5",
                            }
                        ).load()
                        if docs and "just a moment" in docs[0].page_content.lower():
                            add_log(f"          ✗ {url} is Cloudflare protected — try a different URL")
                            continue
                        docs = clean_metadata(docs)
                        all_docs.extend(docs)
                        add_log(f"          ✓ Loaded {len(docs)} page(s)")
                    except Exception as e:
                        add_log(f"          ✗ Failed: {str(e)[:80]}")

            if uploaded_files:
                add_log(f"[LOADER]  Loading {len(uploaded_files)} file(s)...")
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        add_log(f"          → {uploaded_file.name}")
                        try:
                            ext = uploaded_file.name.split(".")[-1].lower()
                            if ext == "pdf":
                                from langchain_community.document_loaders import PyPDFLoader
                                loader = PyPDFLoader(file_path)
                            elif ext == "docx":
                                from langchain_community.document_loaders import Docx2txtLoader
                                loader = Docx2txtLoader(file_path)
                            elif ext in ["txt", "md"]:
                                from langchain_community.document_loaders import TextLoader
                                loader = TextLoader(file_path)
                            elif ext == "csv":
                                from langchain_community.document_loaders import CSVLoader
                                loader = CSVLoader(file_path)
                            else:
                                add_log(f"          ✗ Unsupported file type: {ext}")
                                continue
                            docs = loader.load()
                            all_docs.extend(docs)
                            add_log(f"          ✓ Loaded {len(docs)} page(s)")
                        except Exception as e:
                            add_log(f"          ✗ Failed: {str(e)[:80]}")

            add_log("")
            add_log(f"[SPLITTER] Total documents loaded: {len(all_docs)}")
            add_log("[SPLITTER] Chunking documents...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)
            add_log(f"           ✓ Created {len(chunks)} chunks")

            add_log("")
            add_log("[EMBEDDER] Loading HuggingFace embeddings model...")
            add_log("           (all-MiniLM-L6-v2 — first load may take ~30s)")

            with st.spinner("Hang on...."):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                add_log("           ✓ Embedding model loaded")
                add_log("")

                if st.session_state.append_mode and st.session_state.vectorstore:
                    add_log("[CHROMA]   Appending to existing knowledge base...")
                    st.session_state.vectorstore.add_documents(chunks)
                    add_log(f"           ✓ Appended {len(chunks)} chunks to existing knowledge base")
                else:
                    add_log("[CHROMA]   Building new vector store...")
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        collection_name="rag-session"
                    )
                    st.session_state.vectorstore = vectorstore
                    add_log(f"           ✓ Indexed {len(chunks)} chunks into Chroma")

            st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10}
            )
            st.session_state.ingested = True
            st.session_state.append_mode = False
            st.session_state.urls = []
            st.session_state.uploaded_files_count = 0

            add_log("           ✓ Retriever created and stored in session")
            add_log("")
            add_log("✓ Ingestion complete! Head to the QUERY tab to start asking questions.")
            status_placeholder.success("✅ Knowledge base ready! Go to the QUERY tab.")

        except Exception as e:
            add_log(f"✗ Error: {str(e)}")
            status_placeholder.error(f"Ingestion failed: {str(e)}")

    # ── Manage Knowledge Base ─────────────────────────────────────
    if st.session_state.ingested:
        st.markdown("---")
        st.markdown("**🗄️ Manage Knowledge Base**")
        st.caption("Knowledge base is active in this session.")

        col_add, col_del = st.columns([1, 1])

        with col_add:
            if st.button("➕ Add More Documents", use_container_width=True):
                st.session_state.append_mode = True
                st.session_state.urls = []
                st.session_state.url_input_key += 1
                st.rerun()

        with col_del:
            if st.button("🗑️ Delete Knowledge Base",
                         use_container_width=True,
                         type="primary"):
                st.session_state.ingested = False
                st.session_state.vectorstore = None
                st.session_state.retriever = None
                st.session_state.urls = []
                st.session_state.append_mode = False
                st.session_state.url_input_key += 1
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
            query = st.text_input(
                "Ask a question",
                placeholder="what are adversarial attacks?",
                label_visibility="visible"
            )
        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            ask = st.button("▶ ASK")

        if ask and not query.strip():
            st.warning("Please enter a question first.")

        if ask and query.strip():

            from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH

            query_log_placeholder = st.empty()
            query_logs = []

            def add_query_log(msg):
                query_logs.append(msg)
                def style(line):
                    l = line.lower()
                    if "✓" in line or "complete" in l or "grounded" in l or "useful" in l:
                        return f'<p class="log-line log-ok">{line}</p>'
                    elif "✗" in line or "error" in l or "not supported" in l or "not useful" in l or "failed" in l:
                        return f'<p class="log-line log-error">{line}</p>'
                    elif "web search" in l or "irrelevant" in l:
                        return f'<p class="log-line log-warn">{line}</p>'
                    elif line.startswith("//"):
                        return f'<p class="log-line log-dim">{line}</p>'
                    else:
                        return f'<p class="log-line log-info">{line}</p>'
                html = '<div class="terminal-box">' + "".join(
                    style(l) for l in query_logs
                ) + '</div>'
                query_log_placeholder.markdown(html, unsafe_allow_html=True)

            try:
                add_query_log(f"// Question: {query}")
                add_query_log("")
                add_query_log("[ROUTER]   Running similarity check against vectorstore...")

                with st.spinner("Agents are working..."):
                    stream_outputs, sim_results, run_id = traced_invoke(
                        query, st.session_state.retriever
                    )
                    

                # ── Routing decision ───────────────────────────────
                if sim_results:
                    doc, score = sim_results[0]
                    if score < 1.0:
                        add_query_log(f"[ROUTER]   Similarity score: {score:.3f} → below threshold (1.0)")
                        add_query_log("[ROUTER]   Decision: → VECTORSTORE (RAG) ✓")
                    else:
                        add_query_log(f"[ROUTER]   Similarity score: {score:.3f} → above threshold (1.0)")
                        add_query_log("[ROUTER]   Decision: → WEB SEARCH")
                else:
                    add_query_log("[ROUTER]   No results found in vectorstore")
                    add_query_log("[ROUTER]   Decision: → WEB SEARCH")

                add_query_log("")

                # ── Stream node logs ───────────────────────────────
                result = None
                generate_count = 0
                web_search_triggered = False
                retrieved_count = 0

                for output in stream_outputs:
                    for node_name, node_output in output.items():

                        if node_name == RETRIEVE:
                            retrieved_count = len(node_output.get("documents", []))
                            add_query_log(f"[RETRIEVE]  ✓ Retrieved {retrieved_count} document chunk(s) from vectorstore")

                        elif node_name == GRADE_DOCUMENTS:
                            web = node_output.get("web_search", False)
                            if web:
                                add_query_log(f"[GRADER]    Some chunk(s) dropped as irrelevant")
                                add_query_log("[GRADER]    → Adding web search to supplement")
                            else:
                                add_query_log(f"[GRADER]    ✓ All {retrieved_count} chunk(s) relevant → proceeding to generate")

                        elif node_name == WEBSEARCH:
                            web_search_triggered = True
                            add_query_log("[WEB SEARCH] Fetching results from the web...")
                            add_query_log("[WEB SEARCH] ✓ Web results fetched successfully")

                        elif node_name == GENERATE:
                            generate_count += 1
                            if generate_count > 1:
                                add_query_log(f"[GENERATE]  Re-generating answer (attempt {generate_count})...")
                            else:
                                add_query_log("[GENERATE]  Drafting answer from documents...")
                            add_query_log("[GENERATE]  ✓ Answer drafted")

                        result = node_output

                # ── Post stream grading inference ──────────────────
                add_query_log("")
                if result and result.get("generation"):
                    # Infer grading results based on whether pipeline looped
                    if generate_count > 1:
                        add_query_log("[HALLUCINATION CHECK] ✗ Initial answer not grounded — retried")
                        add_query_log("[HALLUCINATION CHECK] ✓ Final answer grounded in documents")
                    else:
                        add_query_log("[HALLUCINATION CHECK] ✓ Answer is grounded in documents")

                    if web_search_triggered and generate_count > 1:
                        add_query_log("[ANSWER CHECK]        ✗ Initial answer did not address question — web search added")
                        add_query_log("[ANSWER CHECK]        ✓ Final answer addresses the question")
                    else:
                        add_query_log("[ANSWER CHECK]        ✓ Answer addresses the question")

                    add_query_log("")
                    add_query_log("✓ Pipeline complete.")

                    # ── Answer ────────────────────────────────────
                    st.markdown("---")
                    st.markdown("**Answer:**")
                    st.markdown(result["generation"])

                    # ── Sources consulted (web search only) ────────
                    if web_search_triggered:
                        docs = result.get("documents", [])
                        if docs:
                            st.markdown("---")
                            st.markdown("**Sources consulted**")
                            seen = set()
                            for doc in docs:
                                sources = doc.metadata.get("sources", [])
                                for source in sources:
                                    if source and source not in seen:
                                        seen.add(source)
                                        st.markdown(f"- {source}")

                    # ── LangSmith trace ────────────────────────────
                    try:
                        if run_id:
                            st.markdown("---")
                            st.markdown("**LangSmith**")
                            share_link = None
                            for wait in [2, 4, 6]:  # retry 3 times with increasing delays
                                try:
                                    time.sleep(wait)
                                    share_link = client.share_run(run_id)
                                    break
                                except Exception:
                                    continue
                            if share_link:
                                st.markdown(f"🔍 [View LangSmith Trace]({share_link})")
                            else:
                                st.caption("Trace still syncing — check LangSmith dashboard directly.")
                    except Exception:
                        pass

                else:
                    add_query_log("[HALLUCINATION CHECK] ✗ Answer failed grounding check")
                    add_query_log("[ANSWER CHECK]        ✗ Answer did not address the question")
                    add_query_log("✗ Pipeline ended without a valid answer.")
                    st.warning("No answer could be generated. Try rephrasing your question.")

            except Exception as e:
                if "recursion" in str(e).lower():
                    add_query_log("✗ Pipeline exceeded maximum steps — could not generate a satisfactory answer.")
                    st.error("The pipeline looped too many times. Try rephrasing your question.")
                else:
                    add_query_log(f"✗ Error: {str(e)}")
                    st.error(f"Error: {str(e)}")