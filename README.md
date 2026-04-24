# Touch Data Quality Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Quality_ML-F7931E?logo=scikit-learn&logoColor=white)
![Google Sheets](https://img.shields.io/badge/Google_Sheets-Export-34A853?logo=google-sheets&logoColor=white)
![Data Validation](https://img.shields.io/badge/Data_Validation-Research_Ready-0EA5E9)

Data engineering and ML quality pipeline for touch-interaction research data. The system processes raw touch records, validates event sequences, engineers quality features, exports clean datasets, and produces visual reports for downstream research workflows.

## What This Demonstrates

- End-to-end data processing from raw JSON/CSV into analysis-ready tables
- Validation and flagging for sequence and quality issues
- ML-assisted quality assessment and interaction-type classification
- Google Sheets export for collaborator-friendly review
- Interactive HTML reports for exploratory analysis
- Modular project structure with tests, docs, config, scripts, and source packages

## Pipeline

```text
Raw touch data
  -> ingestion and schema handling
  -> validation and quality flags
  -> feature engineering
  -> ML-enhanced quality assessment
  -> CSV / Google Sheets / HTML report outputs
```

## Core Features

| Feature | Description |
|---|---|
| Data processing | Converts raw interaction records into clean tabular outputs |
| Validation | Flags sequence, identifier, and behavioral anomalies |
| ML enhancement | Adds quality scores and interaction-type labels |
| Export | Supports CSV and Google Sheets workflows |
| Visualization | Generates interactive HTML reports |
| Testing | Includes integration, ML, and performance tests |

## Quick Start

```bash
git clone https://github.com/jbanmol/touch-data-quality-pipeline.git
cd touch-data-quality-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Repository Structure

| Path | Purpose |
|---|---|
| `src/core/` | Main data processing engine |
| `src/ml/` | ML quality assessment components |
| `src/export/` | Google Sheets and file export utilities |
| `src/visualization/` | Interactive report generation |
| `tests/` | Integration, ML, and performance checks |
| `docs/` | User and developer documentation |

## Tech Stack

Python, pandas, scikit-learn, Google Sheets API, HTML reports, pytest.

## Professional Focus

This repository is a data-engineering companion to the clinical ML work: it shows the upstream systems thinking needed before modeling can be trusted.
