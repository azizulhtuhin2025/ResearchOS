# # Research Agent v2
# Pipeline: **plan → search (OpenAlex + Semantic Scholar + arXiv + web) → dedupe → rank → extract → verify → report**
#
# - Works with **no API key** (extractive mode).
# - If `ANTHROPIC_API_KEY` is set, Claude plans the sub-questions and writes the synthesis; every citation is still checked against the evidence.
# - Output: `report.md` (cited report) and `evidence.json` (everything the report is based on).

# ## 1. Install

# pip install -q ddgs requests beautifulsoup4 sentence-transformers numpy anthropic

# ## 2. Config

import os, re, json, math, time, datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from bs4 import BeautifulSoup


@dataclass
class Config:
    max_subq: int = 5                # sub-questions in the plan
    per_source: int = 8              # results per scholarly source, per sub-question
    web_results: int = 4             # web pages per sub-question (set 0 to disable web)
    top_docs_per_subq: int = 5       # sources kept per sub-question
    findings_per_doc: int = 2        # key sentences taken per source
    timeout: int = 15
    mailto: str = "your_email@example.com"   # OpenAlex "polite pool"; put your real email
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-5-5"     # used only if ANTHROPIC_API_KEY is set
    w_sem: float = 0.70              # ranking weights: relevance / source quality / recency
    w_quality: float = 0.20
    w_recency: float = 0.10
    agree_threshold: float = 0.60    # similarity at which another source "agrees"
    min_support: float = 0.35        # LLM claims below this vs their cited source get flagged


CFG = Config()
YEAR = datetime.date.today().year

# ## 3. Data model and search sources

@dataclass
class Doc:
    title: str
    text: str
    url: str
    origin: str                 # openalex | semanticscholar | arxiv | web
    year: int | None = None
    venue: str = ""
    citations: int = 0
    doi: str = ""
    kind: str = "web"           # journal | preprint | web
    authors: str = ""
    id: int = 0


KIND_RANK = {"web": 0, "preprint": 1, "journal": 2}
STOP = set("a an the of for to in on and or is are can we be by with how what which that this from as at it its do does into than".split())


def keywords(q, n=8):
    out = []
    for w in re.findall(r"[a-z0-9\-]+", q.lower()):
        if w not in STOP and len(w) > 2 and w not in out:
            out.append(w)
    return out[:n]


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def http_get(url, params=None, headers=None, retries=2):
    h = {"User-Agent": f"research-agent/2.0 (mailto:{CFG.mailto})"}
    if headers:
        h.update(headers)
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=h, timeout=CFG.timeout)
            if r.status_code in (429, 503):
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(1)
    return None


def short_authors(names):
    names = [n for n in names if n]
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


# ---- arXiv -------------------------------------------------------------
def parse_arxiv(xml_text):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    docs = []
    for e in ET.fromstring(xml_text).findall("a:entry", ns):
        title = " ".join(e.findtext("a:title", "", ns).split())
        abstract = " ".join(e.findtext("a:summary", "", ns).split())
        pub = e.findtext("a:published", "", ns)
        if not (title and abstract):
            continue
        docs.append(Doc(title=title, text=abstract, url=e.findtext("a:id", "", ns), origin="arxiv",
                        year=int(pub[:4]) if pub[:4].isdigit() else None, venue="arXiv", kind="preprint",
                        authors=short_authors([a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)])))
    return docs


def search_arxiv(q, n):
    kws = keywords(q, 5)
    for k in dict.fromkeys([len(kws), 3]):          # strict query first, then looser
        if not kws:
            break
        sq = " AND ".join(f"all:{w}" for w in kws[:k])
        r = http_get("https://export.arxiv.org/api/query",
                     {"search_query": sq, "max_results": n, "sortBy": "relevance"})
        if r is None:
            return []
        docs = parse_arxiv(r.text)
        if docs:
            return docs
    return []


# ---- OpenAlex ----------------------------------------------------------
def rebuild_abstract(inv):
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def search_openalex(q, n):
    r = http_get("https://api.openalex.org/works", {
        "search": q, "per-page": n, "filter": "has_abstract:true", "mailto": CFG.mailto,
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,primary_location,authorships"})
    if r is None:
        return []
    docs = []
    for w in r.json().get("results", []):
        abstract = rebuild_abstract(w.get("abstract_inverted_index"))
        title = w.get("title") or ""
        if not (abstract and title):
            continue
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name") or ""
        kind = "preprint" if (src.get("type") == "repository" or not venue) else "journal"
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        docs.append(Doc(title=title, text=abstract, url=w.get("doi") or loc.get("landing_page_url") or w.get("id", ""),
                        origin="openalex", year=w.get("publication_year"), venue=venue,
                        citations=w.get("cited_by_count") or 0, doi=doi, kind=kind,
                        authors=short_authors([(a.get("author") or {}).get("display_name") for a in w.get("authorships", [])])))
    return docs


# ---- Semantic Scholar --------------------------------------------------
def search_s2(q, n):
    key = os.getenv("S2_API_KEY")                    # optional, raises rate limits
    r = http_get("https://api.semanticscholar.org/graph/v1/paper/search",
                 {"query": q, "limit": n, "fields": "title,abstract,year,venue,citationCount,externalIds,url,authors"},
                 headers={"x-api-key": key} if key else None)
    if r is None:
        return []
    docs = []
    for p in r.json().get("data", []) or []:
        if not (p.get("abstract") and p.get("title")):
            continue
        venue = p.get("venue") or ""
        kind = "journal" if venue and "arxiv" not in venue.lower() else "preprint"
        docs.append(Doc(title=p["title"], text=p["abstract"], url=p.get("url") or "", origin="semanticscholar",
                        year=p.get("year"), venue=venue or "preprint", citations=p.get("citationCount") or 0,
                        doi=(p.get("externalIds") or {}).get("DOI", ""), kind=kind,
                        authors=short_authors([a.get("name") for a in p.get("authors", [])])))
    return docs


# ---- Web ---------------------------------------------------------------
def fetch_text(url):
    r = http_get(url, retries=0)
    if r is None or "html" not in r.headers.get("content-type", "").lower():
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "svg", "iframe", "form"]):
        t.extract()
    root = soup.find("article") or soup.find("main") or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    paras = [p for p in paras if len(p.split()) >= 8]
    return " ".join(" ".join(paras).split()[:1800])


def search_web(q, n):
    if n <= 0:
        return []
    try:
        from ddgs import DDGS
        with DDGS() as d:
            hits = list(d.text(q, max_results=n))
    except Exception:
        return []
    docs = []
    for h in hits:
        url = h.get("href") or h.get("url")
        text = fetch_text(url) if url else ""
        if len(text.split()) >= 60:
            docs.append(Doc(title=h.get("title") or url, text=text, url=url, origin="web", kind="web"))
    return docs


def retrieve(query):
    """Run all sources in parallel; a failing source never stops the run."""
    jobs = [(search_openalex, CFG.per_source), (search_s2, CFG.per_source),
            (search_arxiv, CFG.per_source), (search_web, CFG.web_results)]

    def safe(job):
        fn, n = job
        try:
            return fn(query, n)
        except Exception as e:
            print(f"   ! {fn.__name__} failed: {type(e).__name__}")
            return []

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(safe, jobs))
    print("   sources:", {fn.__name__.replace("search_", ""): len(r) for (fn, _), r in zip(jobs, results)})
    return [d for r in results for d in r]


class Store:
    """Global de-duplication by DOI or normalized title; merges metadata of duplicates."""
    def __init__(self):
        self.docs, self.by_key = [], {}

    def add(self, d):
        keys = [k for k in ((d.doi or "").lower(), norm_title(d.title)) if k]
        for k in keys:
            if k in self.by_key:
                ex = self.by_key[k]
                ex.citations = max(ex.citations, d.citations)
                if KIND_RANK[d.kind] > KIND_RANK[ex.kind]:
                    ex.kind, ex.venue = d.kind, d.venue or ex.venue
                ex.doi = ex.doi or d.doi
                ex.year = ex.year or d.year
                if len(d.text) > len(ex.text):
                    ex.text = d.text
                for k2 in keys:
                    self.by_key.setdefault(k2, ex)
                return ex
        d.id = len(self.docs) + 1
        self.docs.append(d)
        for k in keys:
            self.by_key[k] = d
        return d

# ## 4. Embeddings, ranking, extraction

class Embedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedder: {name}")
        self.m, self.cache = SentenceTransformer(name), {}

    def __call__(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            vecs = self.m.encode(new, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=32)
            self.cache.update(zip(new, vecs))
        return np.stack([self.cache[t] for t in texts])


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def passages(d):
    if d.kind != "web":
        return [f"{d.title}. {d.text}"]
    w = d.text.split()
    return [" ".join(w[i:i + 120]) for i in range(0, min(len(w), 720), 120)]


def quality(d):
    base = {"journal": 1.0, "preprint": 0.7, "web": 0.4}[d.kind]
    cites = min(math.log10(d.citations + 1) / 3, 1.0)       # 1000 citations -> 1.0
    return 0.6 * base + 0.4 * cites


def recency(d):
    return math.exp(-max(0, YEAR - d.year) / 5) if d.year else 0.5


def rank(emb, question, docs):
    """Best passage per doc, scored by relevance + source quality + recency."""
    cands = [(d, p) for d in docs for p in passages(d)]
    if not cands:
        return []
    sims = emb([p for _, p in cands]) @ emb([question])[0]
    best = {}
    for (d, p), s in zip(cands, sims):
        if d.id not in best or s > best[d.id]["sim"]:
            best[d.id] = {"doc": d, "passage": p, "sim": float(s)}
    for b in best.values():
        b["score"] = CFG.w_sem * b["sim"] + CFG.w_quality * quality(b["doc"]) + CFG.w_recency * recency(b["doc"])
    return sorted(best.values(), key=lambda b: -b["score"])[:CFG.top_docs_per_subq]


CUES = re.compile(r"\b(we (show|find|propose|demonstrate|prove|introduce|study)|results?|outperform\w*|improv\w*|"
                  r"achiev\w*|reduc\w*|increas\w*|however|limitation\w*|suggest\w*|evidence|significant\w*)\b", re.I)


def extract_findings(emb, question, picks):
    q = emb([question])[0]
    out = []
    for p in picks:
        sents = [s for s in split_sentences(p["passage"]) if 8 <= len(s.split()) <= 60]
        if not sents:
            continue
        sim = emb(sents) @ q
        bonus = np.array([0.05 * bool(CUES.search(s)) + 0.03 * bool(re.search(r"\d", s)) for s in sents])
        for i in np.argsort(-(sim + bonus))[:CFG.findings_per_doc]:
            out.append({"claim": sents[i], "doc": p["doc"], "sim": float(sim[i])})
    return out

# ## 5. Verification: grounding, cross-source agreement, confidence

def support(emb, claim, sentences):
    """How well a claim is backed by a source's sentences (max cosine)."""
    return float((emb(sentences) @ emb([claim])[0]).max()) if sentences else 0.0


def agreeing_sources(emb, claim, doc, picks):
    """Number of OTHER sources with a sentence close to this claim."""
    c, n = emb([claim])[0], 0
    for p in picks:
        if p["doc"].id == doc.id:
            continue
        sents = split_sentences(p["passage"])
        if sents and float((emb(sents) @ c).max()) >= CFG.agree_threshold:
            n += 1
    return n


def confidence(n_agree, doc):
    if n_agree >= 2 or (n_agree == 1 and doc.kind == "journal"):
        return "High"
    return "Medium" if (n_agree == 1 or doc.kind == "journal") else "Low"


def verify_findings(emb, section):
    for f in section["findings"]:
        d = f["doc"]
        f["grounded"] = norm_title(f["claim"]) in norm_title(d.title + " " + d.text)   # verbatim from source
        f["agree"] = agreeing_sources(emb, f["claim"], d, section["picks"])
        f["confidence"] = confidence(f["agree"], d)

# ## 6. Planner and LLM (optional)

PLAN_SYSTEM = ("You plan literature research. Break the user's question into distinct sub-questions that together "
               "cover background, methods, evidence/results, limitations and recent work (as relevant). "
               'Return ONLY a JSON list: [{"question": "...", "search": "3-7 keyword query for academic search"}].')

SYNTH_SYSTEM = ("You are a careful research analyst. Use ONLY the numbered evidence provided. "
                "Every factual sentence must end with citation(s) like [3] or [2][5]. "
                "If the evidence does not answer something, say so plainly. Never invent numbers, papers or authors. "
                "Where sources conflict, state the conflict explicitly.")


def make_llm():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY -> extractive mode (set it for LLM planning + synthesis).")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        print("anthropic SDK unavailable -> extractive mode:", e)
        return None

    def call(system, user, max_tokens=2500):
        msg = client.messages.create(model=CFG.llm_model, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if b.type == "text")
    print(f"LLM mode: {CFG.llm_model}")
    return call


def heuristic_plan(q):
    kw = " ".join(keywords(q, 6))
    facets = [("Background and definitions", "survey overview"), ("Main methods and approaches", "methods approach"),
              ("Empirical evidence and results", "results experiments"),
              ("Limitations, criticisms and open problems", "limitations challenges"),
              ("Most recent advances", "recent advances")]
    return [{"question": f"{label}: {q}", "search": f"{kw} {sfx}"} for label, sfx in facets[:CFG.max_subq]]

# ## 7. The agent

def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


class ResearchAgent:
    def __init__(self, embedder=None, llm="auto"):
        self.emb = embedder or Embedder(CFG.embed_model)
        self.llm = make_llm() if llm == "auto" else llm

    def plan(self, question):
        if self.llm:
            try:
                raw = self.llm(PLAN_SYSTEM, f"Research question: {question}\nReturn {CFG.max_subq} items.", 800)
                items = json.loads(raw.replace("```json", "").replace("```", "").strip())
                plan = [{"question": i["question"], "search": i["search"]} for i in items if i.get("question") and i.get("search")]
                if plan:
                    return plan[:CFG.max_subq]
            except Exception as e:
                print("   planner fallback:", type(e).__name__)
        return heuristic_plan(question)

    def run(self, question, out_dir=".", progress=None):
        t0, store, sections, seen = time.time(), Store(), [], set()
        plan = self.plan(question)
        for i, sq in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] {sq['question']}")
            if progress:
                progress(i, len(plan), sq["question"])             # lets a UI (Streamlit) show live progress
            docs = list({d.id: d for d in (store.add(x) for x in retrieve(sq["search"]))}.values())
            picks = rank(self.emb, sq["question"], docs)
            finds = [f for f in extract_findings(self.emb, sq["question"], picks) if f["claim"] not in seen]
            seen.update(f["claim"] for f in finds)                 # never repeat a claim across sub-questions
            sec = {"sq": sq, "picks": picks, "findings": finds}
            verify_findings(self.emb, sec)
            sections.append(sec)

        ref = {}                                           # doc.id -> reference number
        for sec in sections:
            for p in sec["picks"]:
                ref.setdefault(p["doc"].id, len(ref) + 1)
        if not ref:
            print("No sources retrieved. Check your internet connection / query.")
            return None

        body, stats = None, {"claims_checked": 0, "claims_flagged": 0}
        if self.llm:
            try:
                body = self.synthesize_llm(question, sections, ref, stats)
            except Exception as e:
                print("   synthesis fallback:", type(e).__name__, e)
        if body is None:
            body = self.synthesize_extractive(sections, ref)

        report = self.render(question, body, store, sections, ref, stats, time.time() - t0)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(out_dir, "evidence.json"), "w", encoding="utf-8") as f:
            json.dump({"question": question, "plan": plan,
                       "sources": {ref[d.id]: {k: getattr(d, k) for k in ("title", "url", "year", "venue", "kind", "citations", "doi", "origin")}
                                   for d in store.docs if d.id in ref},
                       "findings": [{"subquestion": s["sq"]["question"], "claim": x["claim"], "ref": ref[x["doc"].id],
                                     "grounded": x["grounded"], "agreeing_sources": x["agree"], "confidence": x["confidence"]}
                                    for s in sections for x in s["findings"]]}, f, indent=2, ensure_ascii=False)
        print(f"Saved report.md and evidence.json in '{out_dir}' ({time.time() - t0:.0f}s)")
        return {"report": report, "sections": sections, "ref": ref, "store": store, "stats": stats,
                "plan": plan, "secs": time.time() - t0, "question": question}

    # ---- synthesis: extractive (no API key) ----
    def synthesize_extractive(self, sections, ref):
        order = {"High": 0, "Medium": 1, "Low": 2}
        allf = sorted((x for s in sections for x in s["findings"]), key=lambda x: (order[x["confidence"]], -x["sim"]))
        lines = ["## Key evidence (extractive summary)", ""]
        lines += [f"- {x['claim']} [{ref[x['doc'].id]}]" for x in allf[:4]]
        lines += ["", "## Findings by sub-question", ""]
        for i, s in enumerate(sections, 1):
            lines.append(f"### {i}. {s['sq']['question']}")
            if not s["findings"]:
                lines.append("_No additional evidence beyond earlier sections._")
            for x in s["findings"]:
                lines.append(f"- {x['claim']} [{ref[x['doc'].id]}] — *{x['confidence']} confidence "
                             f"({x['agree']} other source(s) agree; {x['doc'].kind})*")
            lines.append("")
        return "\n".join(lines)

    # ---- synthesis: Claude, then every citation is checked ----
    def synthesize_llm(self, question, sections, ref, stats):
        ev, ref_sents = [], {}
        for i, s in enumerate(sections, 1):
            ev.append(f"### Sub-question {i}: {s['sq']['question']}")
            for p in s["picks"]:
                d, n = p["doc"], ref[p["doc"].id]
                ev.append(f"[{n}] {d.title} ({d.year or 'n.d.'}, {d.venue or d.origin}, {d.kind})\n{p['passage'][:1200]}")
                ref_sents.setdefault(n, []).extend(split_sentences(p["passage"]))
        user = (f"Research question: {question}\n\nEVIDENCE:\n" + "\n\n".join(ev) +
                "\n\nWrite a markdown report with exactly these parts:\n"
                "## Executive summary (3-5 sentences)\n"
                "## Findings (one ### heading per sub-question, 2-4 bullet claims each)\n"
                "## Disagreements and gaps\n"
                "Cite every claim with [n]. Do not add a references list.")
        text = self.llm(SYNTH_SYSTEM, user, 3000)
        out = []
        for line in text.splitlines():
            ids = {int(x) for x in re.findall(r"\[(\d+)\]", line)}
            if ids and not line.lstrip().startswith("#"):
                stats["claims_checked"] += 1
                claim = re.sub(r"\[\d+\]", "", line).strip(" -*")
                if not ids <= set(ref_sents):
                    line += "  ⚠️ *invalid citation*"
                    stats["claims_flagged"] += 1
                elif max(support(self.emb, claim, ref_sents[n]) for n in ids) < CFG.min_support:
                    line += "  ⚠️ *weak support in cited source*"
                    stats["claims_flagged"] += 1
            out.append(line)
        return "\n".join(out)

    def render(self, question, body, store, sections, ref, stats, secs):
        origin = {}
        for d in store.docs:
            origin[d.origin] = origin.get(d.origin, 0) + 1
        n_find = sum(len(s["findings"]) for s in sections)
        conf = {c: sum(1 for s in sections for x in s["findings"] if x["confidence"] == c) for c in ("High", "Medium", "Low")}
        mode = f"LLM-assisted ({CFG.llm_model})" if self.llm and stats["claims_checked"] else "extractive (no LLM)"
        L = [f"# Research report: {question}", "",
             f"*Generated {datetime.date.today()} · mode: {mode} · {len(store.docs)} unique sources retrieved "
             f"({', '.join(f'{k} {v}' for k, v in origin.items())}) · {secs:.0f}s*", "", body, "",
             "## Evidence quality", "",
             f"- Extracted findings: {n_find} (High {conf['High']}, Medium {conf['Medium']}, Low {conf['Low']} confidence)",
             "- Confidence = how many *other* retrieved sources contain a similar statement, plus source type (journal > preprint > web).",
             "- Every extracted finding is a verbatim sentence from its source." if not stats["claims_checked"] else
             f"- LLM claims checked against their cited source: {stats['claims_checked']}; flagged: {stats['claims_flagged']} (marked ⚠️)."]
        L += ["", "## Sources", "", "| # | Title | Year | Venue / type | Citations |", "|---|---|---|---|---|"]
        for d in sorted((d for d in store.docs if d.id in ref), key=lambda d: ref[d.id]):
            title = f"[{esc(d.title)}]({d.url})" if d.url else esc(d.title)
            L.append(f"| {ref[d.id]} | {title} | {d.year or '-'} | {esc(d.venue or d.origin)} ({d.kind}) | {d.citations or '-'} |")
        L += ["", "## Limitations", "",
              "- Scholarly sources are judged from **abstracts only**; read the papers before relying on any number.",
              "- Web pages are lower-quality evidence and are weighted down, not excluded.",
              "- Agreement between sources is measured by embedding similarity, not by logical entailment. Contradiction detection is only done in LLM mode, inside the 'Disagreements and gaps' section."]
        return "\n".join(L)

# ## 8. Run it

if __name__ == "__main__":
    QUESTION = "Can we identify the minimal parameters required for learning a new task?"
    agent = ResearchAgent()
    result = agent.run(QUESTION, out_dir="output")
    if result:
        try:
            from IPython.display import Markdown, display
            display(Markdown(result["report"]))
        except Exception:
            print(result["report"])
