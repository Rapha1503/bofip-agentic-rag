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
DEEPSEEK_API_KEY=<votre-cle-api-deepseek>
```

## Runtime Artifacts

Download the full-corpus runtime artifacts:

```powershell
python scripts/download_artifacts.py
python scripts/check_setup.py --deep --skip-models
```

A local machine with the model cache can also place these files manually under `data/interim/`:

- `raw_docs.jsonl`
- `chunks.jsonl`
- `doc_dense_cache.npy`
- `chunk_dense_cache.npy`

## Run The App

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The app uses BYOK: the API key is read from `.env.local` or entered in the UI.

## What To Check

Ask a BOFiP-style fiscal question, for example:

```text
Quel taux de TVA pour la pose d'une pompe à chaleur chez un particulier ?
```

A healthy run should show:

- a status: `Réponse sourcée`, `Preuve bloquante manquante`, or `Preuve insuffisante`;
- coverage metrics and axes;
- a visible `Parcours agentique réel` with iterations, domain prefix, retrieval count, and reformulation if triggered;
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
