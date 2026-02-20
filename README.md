# Fraud GenAI Explainer (MVP)

This project trains a baseline fraud scorer (LightGBM) on the public credit card dataset,
enriches it with synthetic entity fields (merchant/device/IP/geo), computes SHAP drivers,
builds **Evidence JSON**, and generates two grounded explanations (analyst + plain language).

## Setup

1) Download the public dataset and place it here:
\\\
data/creditcard.csv
\\\

2) Create venv + install:
\\\powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
\\\

3) Run:
\\\powershell
python .\run.py
\\\

Outputs:
- \outputs/alerts_with_evidence.jsonl\
- \outputs/llm_reports.jsonl\

> Note: the LLM call is stubbed in \src/llm.py\ (deterministic template). Replace it with a real API call later.
