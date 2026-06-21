# Local Demo Guide

## Install

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Add a provider key to `.env.local`:

```text
DEEPSEEK_API_KEY=sk-...
```

## Runtime Artifacts

Download the full-corpus runtime artifacts:

```powershell
python scripts/download_artifacts.py
python scripts/check_setup.py --deep --skip-models
```

A local machine with the model cache can also place these files manually under `data/interim/`:

- `raw_docs_sample_5666.jsonl`
- `chunks_section_window_sample_5666.jsonl`
- `doc_dense_cache_5666_sections_firstpara_e5large.npy`
- `chunk_dense_cache_5666_full_e5large.npy`

## Run The App

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The app uses BYOK: the API key is read from `.env.local` or entered in the UI.

## What To Check

Ask a BOFiP-style fiscal question, for example:

```text
Quel taux de TVA pour la pose d'une pompe ? chaleur chez un particulier ?
```

A healthy run should show:

- a status: `R?ponse sourc?e`, `R?ponse partielle`, or `Preuve insuffisante`;
- coverage metrics and axes;
- a visible `Parcours agentique r?el` with iterations, domain prefix, retrieval count, and reformulation if triggered;
- BOFiP source cards.

## Tests

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## Evaluation

```powershell
$env:PYTHONPATH='src'
python scripts/eval_full.py --limit 3
python scripts/eval_full.py --resume
```

The evaluation calls the configured LLM provider and may take time on CPU.
