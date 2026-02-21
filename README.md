# Credit Card Fraud Detection with Explainable AI

This project trains a baseline fraud scorer LightGBM (Light Gradient-Boosting Machine) on the public credit card dataset,
computes SHAP drivers, and generates grounded explanations (analyst + plain language). An explainable fraud alert workflow is established, where every flagged transaction is accompanied by structured evidence and traceable reasoning, improving transparency for investigation and audit purposes.

The goal is to demonstrate how traditional ML scoring + explainability can be combined
with structured evidence and GenAI to support credit card fraud investigations.

---

## Setup


### 1) Create Virtual Environment + Install Dependencies

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you're on macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

---

### 2) Run the Project

```powershell
python .\run.py
```

---

## Outputs

Running the pipeline will generate:

- `outputs/alerts_with_evidence.jsonl`
- `outputs/llm_reports.jsonl`

These contain:
- Model scores
- Structured evidence objects
- Generated explanation reports

---

## Notes 

- The LLM call is currently stubbed in `src/llm.py` (deterministic template output).
- Replace the stub with a real API call if you want live GenAI-generated explanations.
- Performance tuning, threshold calibration, and monitoring are intentionally minimal.
