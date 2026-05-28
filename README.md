# Cyber Assurance Predictive Hybrid Platform

No LLM. No Torch. No Transformers.

This version includes:
- Broad cybersecurity control taxonomy
- Automated control-domain and sub-control classification
- Constraint-based MITRE ATT&CK mapping
- Optional MITRE STIX JSON upload/import
- CTI / Kaggle past incident enrichment
- Risk Score, Breach Probability, Risk Gauge, Threat Classification
- 30/60/90 day future posture forecasting
- Management-friendly summaries
- Finding → CTI incident mapping
- Finding → MITRE technique mapping with confidence and explanation
- Excel export with multiple sheets

## Run
```powershell
cd cyber_assurance_predictive_hybrid_final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Inputs
Upload a risk register CSV/XLSX. Recommended columns:
- Finding ID
- Title
- Description
- Severity
- Status
- Owner
- Business Unit
- Domain
- Technology
- Root Cause
- Created Date
- Due Date
- Closure Date
- Internet Facing / Exposure / Environment / Asset Criticality if available

Optional CTI dataset columns:
- Incident Name
- Description
- Threat Type
- Industry
- Technique ID
- Technique Name
- Source
- Year
- Keywords

Optional MITRE file:
- enterprise-attack.json from https://github.com/mitre-attack/attack-stix-data

## MITRE ATT&CK STIX Upload
The final build includes a dedicated sidebar uploader: **Optional: Upload MITRE ATT&CK STIX JSON**.
Upload `enterprise-attack.json` from the official MITRE ATT&CK STIX repository. The application uses the uploaded file to enrich official technique names, tactics, platforms, detection text, and URLs while keeping conservative rule-based validation.
