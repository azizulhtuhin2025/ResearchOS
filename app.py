"""Streamlit dashboard for research_agent_v2.py

Run:   streamlit run app.py
Needs: research_agent_v2.py in the same folder.
"""
import os
import sys
import json
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research_agent_v2 as ra  # noqa: E402

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

EXAMPLES = [
    "Can we identify the minimal parameters required for learning a new task?",
    "How do physics-informed neural networks compare with classical solvers for PDEs?",
    "What are the main approaches to forecasting solar power output with deep learning?",
]


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embedder(name: str):
    return ra.Embedder(name)


def build_tables(res):
    """Turn the agent result into two DataFrames: findings and sources."""
    ref, store = res["ref"], res["store"]
    docs = {d.id: d for d in store.docs}
    findings = []
    for s in res["sections"]:
        for x in s["findings"]:
            d = x["doc"]
            findings.append({
                "Sub-question": s["sq"]["question"],
                "Finding": x["claim"],
                "Ref": ref[d.id],
                "Source": d.title,
                "Type": d.kind,
                "Year": d.year,
                "Confidence": x["confidence"],
                "Agreeing sources": x["agree"],
                "Verbatim": x["grounded"],
            })
    sources = []
    for doc_id, n in sorted(ref.items(), key=lambda kv: kv[1]):
        d = docs[doc_id]
        sources.append({
            "Ref": n, "Title": d.title, "Authors": d.authors, "Year": d.year, "Venue": d.venue or d.origin,
            "Type": d.kind, "Citations": d.citations, "DOI": d.doi, "Link": d.url, "Found via": d.origin,
        })
    return pd.DataFrame(findings), pd.DataFrame(sources)


def run_agent(question, settings, api_key):
    for k, v in settings.items():
        setattr(ra.CFG, k, v)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    agent = ra.ResearchAgent(embedder=get_embedder(ra.CFG.embed_model), llm="auto")
    bar = st.progress(0.0, text="Planning...")

    def progress(i, n, sub):
        bar.progress((i - 1) / n, text=f"Step {i}/{n}: {sub[:90]}")

    res = agent.run(question, out_dir="output", progress=progress)
    bar.progress(1.0, text="Done")
    if res:
        res["llm_used"] = bool(agent.llm and res["stats"]["claims_checked"])
    return res


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
lit_mode = st.sidebar.toggle("Literature-review mode", value=False,
                             help="Scholarly sources only, wider search, longer reading list.")
api_key = st.sidebar.text_input("Anthropic API key (optional)", type="password",
                                value=os.getenv("ANTHROPIC_API_KEY", ""),
                                help="With a key, Claude plans sub-questions and writes the synthesis. "
                                     "Every claim is still checked against its cited source.")
mailto = st.sidebar.text_input("Your email (OpenAlex polite pool)", value=ra.CFG.mailto)
max_subq = st.sidebar.slider("Sub-questions", 2, 8, 5)
per_source = st.sidebar.slider("Results per source", 5, 40, 30 if lit_mode else 8)
top_docs = st.sidebar.slider("Sources kept per sub-question", 3, 20, 15 if lit_mode else 5)
web_results = st.sidebar.slider("Web pages per sub-question", 0, 8, 0 if lit_mode else 4,
                                disabled=lit_mode)
settings = dict(max_subq=max_subq, per_source=per_source, top_docs_per_subq=top_docs,
                web_results=0 if lit_mode else web_results, mailto=mailto)

if "runs" not in st.session_state:
    st.session_state.runs = []

if st.session_state.runs:
    st.sidebar.divider()
    st.sidebar.subheader("History")
    labels = [f"{i + 1}. {r['question'][:45]}" for i, r in enumerate(st.session_state.runs)]
    pick = st.sidebar.selectbox("Open earlier run", range(len(labels)), index=len(labels) - 1,
                                format_func=lambda i: labels[i])
else:
    pick = None

# ---------------------------------------------------------------- main
st.title("🔎 Research Agent")
st.caption("plan → search (OpenAlex · Semantic Scholar · arXiv · web) → dedupe → rank → extract → verify → report")

if "q" not in st.session_state:
    st.session_state.q = EXAMPLES[0]

with st.form("ask"):
    question = st.text_area("Research question", value=st.session_state.q, height=90)
    go = st.form_submit_button("Run research", type="primary")

st.caption("Try: " + " · ".join(f"_{e}_" for e in EXAMPLES[1:]))

if go and question.strip():
    st.session_state.q = question.strip()
    t = time.time()
    with st.spinner("Researching..."):
        try:
            res = run_agent(question.strip(), settings, api_key.strip())
        except Exception as e:  # keep the UI alive on network / model errors
            st.error(f"Run failed: {type(e).__name__}: {e}")
            res = None
    if res:
        st.session_state.runs.append(res)
        pick = len(st.session_state.runs) - 1
    elif res is None and not st.session_state.runs:
        st.warning("No sources were retrieved. Check your internet connection and try again.")

if pick is not None:
    res = st.session_state.runs[pick]
    findings, sources = build_tables(res)
    conf = findings["Confidence"].value_counts() if len(findings) else pd.Series(dtype=int)

    st.subheader(res["question"])
    c = st.columns(6)
    c[0].metric("Sources used", len(sources))
    c[1].metric("Findings", len(findings))
    c[2].metric("High confidence", int(conf.get("High", 0)))
    c[3].metric("Low confidence", int(conf.get("Low", 0)))
    c[4].metric("Flagged claims", res["stats"]["claims_flagged"] if res["llm_used"] else "n/a")
    c[5].metric("Time", f"{res['secs']:.0f}s")
    st.caption("Mode: " + (f"LLM-assisted ({ra.CFG.llm_model})" if res["llm_used"] else "extractive (no LLM)"))

    tab_report, tab_find, tab_src, tab_plan, tab_dl = st.tabs(
        ["📄 Report", "🧩 Findings", "📚 Sources", "🗺️ Plan", "⬇️ Export"])

    with tab_report:
        st.markdown(res["report"])

    with tab_find:
        f1, f2 = st.columns([1, 2])
        want = f1.multiselect("Confidence", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        subs = f2.multiselect("Sub-question", sorted(findings["Sub-question"].unique()) if len(findings) else [],
                              default=sorted(findings["Sub-question"].unique()) if len(findings) else [])
        view = findings[findings["Confidence"].isin(want) & findings["Sub-question"].isin(subs)] if len(findings) else findings
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={"Finding": st.column_config.TextColumn(width="large"),
                                    "Source": st.column_config.TextColumn(width="medium")})
        st.caption("Confidence = how many other retrieved sources say something similar, plus source type "
                   "(journal > preprint > web). Findings come from abstracts, so verify in the full paper.")

    with tab_src:
        g1, g2 = st.columns(2)
        if len(sources):
            by_year = sources.dropna(subset=["Year"]).astype({"Year": int}).groupby("Year").size()
            g1.markdown("**Sources by year**")
            g1.bar_chart(by_year)
            g2.markdown("**Sources by type**")
            g2.bar_chart(sources.groupby("Type").size())
        st.dataframe(sources, width="stretch", hide_index=True,
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="open"),
                                    "Title": st.column_config.TextColumn(width="large")})

    with tab_plan:
        for i, s in enumerate(res["sections"], 1):
            with st.expander(f"{i}. {s['sq']['question']}", expanded=False):
                st.markdown(f"**Search query:** `{s['sq']['search']}`")
                for p in s["picks"]:
                    d = p["doc"]
                    st.markdown(f"- [{d.title}]({d.url}) — {d.year or 'n.d.'}, {d.kind}, "
                                f"relevance {p['sim']:.2f}, score {p['score']:.2f}")

    with tab_dl:
        evidence = json.dumps({
            "question": res["question"], "plan": res["plan"],
            "sources": sources.to_dict("records"), "findings": findings.to_dict("records"),
        }, indent=2, ensure_ascii=False, default=str)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("report.md", res["report"], "report.md", "text/markdown")
        d2.download_button("evidence.json", evidence, "evidence.json", "application/json")
        d3.download_button("sources.csv", sources.to_csv(index=False), "sources.csv", "text/csv")
        d4.download_button("findings.csv", findings.to_csv(index=False), "findings.csv", "text/csv")
else:
    st.info("Enter a research question and press **Run research**. First run downloads the embedding model (~90 MB).")
