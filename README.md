# Research Agent v2

A literature-research agent that turns a question into a cited, evidence-checked report.

**Pipeline:** plan → search (OpenAlex · Semantic Scholar · arXiv · web) → dedupe → rank → extract → verify → report

- Works with **no API key** (extractive mode: findings are verbatim sentences from sources).
- With an `ANTHROPIC_API_KEY`, Claude plans the sub-questions and writes the synthesis. Every citation is still checked against the cited source, and weak or invalid claims are flagged with ⚠️.
- Ships with a **Streamlit dashboard** (`app.py`) and a plain **script/CLI** mode (`research_agent_v2.py`).

## Files

| File | Purpose |
|---|---|
| `research_agent_v2.py` | The agent: search, ranking, extraction, verification, report generation |
| `app.py` | Streamlit dashboard on top of the agent |
| `requirements.txt` | Python dependencies |
| `output/` | Created on run; holds `report.md` and `evidence.json` |

## Requirements

- Python 3.10+ (the code uses `int | None` type syntax; tested target is 3.13)
- Internet connection
- ~90 MB free disk for the embedding model (`all-MiniLM-L6-v2`), downloaded automatically on first run

## Setup

```bash
# optional but recommended: virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Optional environment variables:

```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # enables LLM planning + synthesis
$env:S2_API_KEY = "..."                 # optional, raises Semantic Scholar rate limits

# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
export S2_API_KEY="..."
```

Also set your real email in `Config.mailto` (in `research_agent_v2.py`) or in the dashboard sidebar. OpenAlex uses it for its "polite pool" (faster, more reliable responses).

## Usage

### Dashboard

```bash
streamlit run app.py
```

Enter a research question and press **Run research**. The dashboard has five tabs:

- **Report**: the full cited report
- **Findings**: filterable table with confidence, agreeing sources, and verbatim-check flag
- **Sources**: reading list with year/type charts, DOIs, and links
- **Plan**: sub-questions, search queries, and the sources picked for each
- **Export**: download `report.md`, `evidence.json`, `sources.csv`, `findings.csv`

Sidebar settings:

| Setting | Effect |
|---|---|
| Literature-review mode | Scholarly sources only, wider search, longer reading list |
| Anthropic API key | Enables LLM mode (can also be set via environment variable) |
| Sub-questions | Number of sub-questions in the plan (2–8) |
| Results per source | Results fetched from each scholarly source per sub-question |
| Sources kept per sub-question | How many top-ranked sources are used |
| Web pages per sub-question | Web results to fetch (disabled in literature-review mode) |

Earlier runs in the session are kept under **History** in the sidebar.

### Script

```bash
python research_agent_v2.py
```

Edit `QUESTION` at the bottom of the file, or use it as a library:

```python
import research_agent_v2 as ra

ra.CFG.max_subq = 4
ra.CFG.web_results = 0          # scholarly sources only

agent = ra.ResearchAgent()
result = agent.run("How do physics-informed neural networks compare with classical solvers for PDEs?",
                   out_dir="output")
print(result["report"])
```

## How it works

1. **Plan**: break the question into sub-questions with short academic search queries (Claude if a key is set, otherwise a built-in heuristic: background, methods, evidence, limitations, recent advances).
2. **Search**: query OpenAlex, Semantic Scholar, arXiv, and the web (DuckDuckGo via `ddgs`) in parallel. A failing source never stops the run.
3. **Dedupe**: global de-duplication by DOI or normalized title; metadata from duplicates is merged.
4. **Rank**: best passage per source scored by `0.70 × relevance + 0.20 × source quality + 0.10 × recency` (embedding similarity; quality uses venue type and citation count).
5. **Extract**: pick the most relevant sentences per source, favoring result-style and numeric statements.
6. **Verify**:
   - *Grounding*: is the finding verbatim in the source?
   - *Agreement*: how many other sources contain a similar sentence?
   - *Confidence*: High / Medium / Low from agreement plus source type (journal > preprint > web).
   - In LLM mode, every cited claim is checked: invalid citation numbers and claims with weak similarity to their cited source are flagged.
7. **Report**: markdown report with evidence-quality summary, sources table, and limitations.

## Configuration

All defaults live in the `Config` dataclass in `research_agent_v2.py`:

| Field | Default | Meaning |
|---|---|---|
| `max_subq` | 5 | Sub-questions in the plan |
| `per_source` | 8 | Results per scholarly source, per sub-question |
| `web_results` | 4 | Web pages per sub-question (0 disables web) |
| `top_docs_per_subq` | 5 | Sources kept per sub-question |
| `findings_per_doc` | 2 | Key sentences taken per source |
| `llm_model` | `claude-sonnet-5-5` | Model used when an API key is set |
| `w_sem` / `w_quality` / `w_recency` | 0.70 / 0.20 / 0.10 | Ranking weights |
| `agree_threshold` | 0.60 | Similarity at which another source "agrees" |
| `min_support` | 0.35 | LLM claims below this vs. their cited source are flagged |

## Output

- `output/report.md`: the cited report
- `output/evidence.json`: question, plan, sources, and every finding with its reference, grounding flag, agreement count, and confidence

## Limitations

- Scholarly sources are judged from **abstracts only**. Read the papers before relying on any number.
- Web pages are lower-quality evidence; they are weighted down, not excluded.
- Agreement is measured by embedding similarity, not logical entailment. Contradiction detection only happens in LLM mode, in the "Disagreements and gaps" section.
- Public APIs rate-limit; if a source returns nothing, rerun later or add an `S2_API_KEY`.
- Web search results depend on `ddgs` and may be empty if the search backend blocks the request.

## Troubleshooting

- **"No sources were retrieved"**: check your internet connection and retry.
- **First run is slow**: the embedding model is downloading (~90 MB); later runs reuse the cache.
- **Streamlit `width="stretch"` error**: upgrade Streamlit (`pip install -U "streamlit>=1.50"`).
- **LLM mode not active**: confirm `ANTHROPIC_API_KEY` is set (or pasted in the sidebar) and the `anthropic` package is installed. The dashboard shows the active mode under the metrics row.
