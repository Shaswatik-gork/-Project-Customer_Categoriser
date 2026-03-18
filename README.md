<div align="center">

<br/>

```
 ██████╗██╗   ██╗███████╗████████╗ ██████╗ ███╗   ███╗███████╗██████╗ 
██╔════╝██║   ██║██╔════╝╚══██╔══╝██╔═══██╗████╗ ████║██╔════╝██╔══██╗
██║     ██║   ██║███████╗   ██║   ██║   ██║██╔████╔██║█████╗  ██████╔╝
██║     ██║   ██║╚════██║   ██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██╔══██╗
╚██████╗╚██████╔╝███████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗██║  ██║
 ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
                                                                         
           ██████╗ █████╗ ████████╗███████╗ ██████╗  ██████╗ ██╗
          ██╔════╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝ ██╔═══██╗██║
          ██║     ███████║   ██║   █████╗  ██║  ███╗██║   ██║██║
          ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██║   ██║╚═╝
          ╚██████╗██║  ██║   ██║   ███████╗╚██████╔╝╚██████╔╝██╗
           ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝
```

### Production-Grade ML Segmentation System

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-customer--categoriser.onrender.com-0ea5e9?style=for-the-badge)](https://customer-categoriser.onrender.com)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-S3-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat-square&logo=render&logoColor=white)

</div>

---

## What Is This?

This is a **fully automated, end-to-end ML system** — not a Jupyter notebook, not a toy demo. It ingests real data from MongoDB, validates it, trains a hybrid ML model, evaluates it, versions it to S3, and serves predictions via a FastAPI endpoint — all without manual intervention.

> **Hybrid Intelligence:** Instead of directly classifying customers, the system first discovers hidden structure via unsupervised clustering (KMeans), then uses those cluster assignments to supercharge a supervised classifier (Logistic Regression). This two-stage approach consistently outperforms single-model classification on segmentation tasks.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           TRAINING PIPELINE          │
                        └─────────────────────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
   ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
   │  Data Ingestion │        │ Data Validation │        │ Transformation  │
   │                 │──────▶│                 │──────▶│                 │
   │  MongoDB Atlas  │        │  Schema + Drift │        │ Feature Eng.   │
   └─────────────────┘        └─────────────────┘        └────────┬────────┘
                                                                   │
                              ┌────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────────┐
            │                 │                      │
            ▼                 ▼                      ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │     KMeans      │ │    Logistic     │ │   Evaluation    │
   │   Clustering    │─│   Regression   │─│  F1 / P / R     │
   │ (Unsupervised)  │ │  (Supervised)  │ │  Accept / Fail  │
   └─────────────────┘ └─────────────────┘ └────────┬────────┘
                                                     │ if accepted
                                                     ▼
                                           ┌─────────────────┐
                                           │   Model Pusher  │
                                           │   → AWS S3      │
                                           │   (Versioned)   │
                                           └────────┬────────┘
                                                    │
                        ┌───────────────────────────┘
                        │
                        ▼
            ┌─────────────────────────────────────┐
            │         PREDICTION PIPELINE          │
            │                                      │
            │   FastAPI ──▶ Load latest model      │
            │           ──▶ Predict category       │
            │           ──▶ Return response        │
            └─────────────────────────────────────┘
```

---

## Project Structure

```
customer_categoriser/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # MongoDB → raw DataFrame
│   │   ├── data_validation.py      # Schema checks, drift detection
│   │   ├── data_transformation.py  # Feature engineering + cluster labels
│   │   ├── model_trainer.py        # KMeans → Logistic Regression
│   │   ├── model_evaluation.py     # F1, Precision, Recall + accept/reject
│   │   └── model_pusher.py         # Upload to AWS S3 (versioned)
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Orchestrates full training run
│   │   └── prediction_pipeline.py  # Loads latest model, serves prediction
│   │
│   ├── configuration/              # Config classes and constants
│   ├── entity/                     # Dataclasses for artifacts and configs
│   ├── cloud_storage/              # AWS S3 connector abstraction
│   └── artifact/                   # Timestamped run outputs (auto-created)
│
├── app.py                          # FastAPI entry point
├── requirements.txt
└── .env.example                    # Template for required secrets
```

---

## ML Strategy

### Stage 1 — Unsupervised (KMeans)

```
Raw Customer Data
       │
       ▼
  ┌─────────┐
  │  KMeans │  → Discovers hidden segments in unlabelled data
  └─────────┘
       │
       ▼
  cluster_id  ← New feature added to the dataset
```

### Stage 2 — Supervised (Logistic Regression)

```
Original Features + cluster_id
              │
              ▼
    ┌──────────────────┐
    │Logistic Regression│  → Learns boundaries enriched by cluster structure
    └──────────────────┘
              │
              ▼
     Customer Category  ← Final prediction
```

**Why hybrid?** Raw features alone may not separate customer groups cleanly. Cluster labels encode higher-level structure that the supervised model can exploit — it's essentially a form of unsupervised feature engineering.

---

## Automated Model Lifecycle

Every training run is fully self-contained and hands-free:

```
train_pipeline.run()
        │
        ├── 1. Creates timestamped artifact/ folder
        ├── 2. Ingests data from MongoDB
        ├── 3. Validates schema and checks for drift
        ├── 4. Transforms features (incl. cluster labels)
        ├── 5. Trains KMeans → Logistic Regression
        ├── 6. Evaluates: F1 / Precision / Recall
        │
        ├── [PASS] ──▶ Save locally + upload to S3
        │                       │
        │               FastAPI auto-loads latest ←──┐
        │               No restart needed            │
        │                                            │
        └── [FAIL] ──▶ Model discarded              ┘
                        Pipeline logs reason
```

---

## Evaluation

| Metric    | Description                                  |
|-----------|----------------------------------------------|
| F1 Score  | Harmonic mean of precision and recall        |
| Precision | How many predicted positives are correct     |
| Recall    | How many actual positives were caught        |

Models must pass an **automated acceptance threshold** before being pushed to production. Failed models are discarded — they never reach S3 or the API.

---

## Tech Stack

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| API             | FastAPI + Uvicorn                 |
| Templates       | Jinja2 + CORS Middleware          |
| ML              | scikit-learn (KMeans, LogReg)     |
| Data            | NumPy, Pandas                     |
| Database        | MongoDB Atlas                     |
| Model Storage   | AWS S3 (auto-versioned)           |
| Deployment      | Render (GitHub auto-deploy)       |

---

## Environment Setup

Copy `.env.example` and populate the following variables. **Never commit real values to Git.**

```bash
cp .env.example .env
```

| Variable                | Purpose                        |
|-------------------------|-------------------------------|
| `MONGO_DB_URL`          | MongoDB Atlas connection string |
| `AWS_ACCESS_KEY_ID`     | S3 write credentials           |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key                  |
| `AWS_DEFAULT_REGION`    | S3 bucket region               |

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/Shaswatik-gork/-Project-Customer_Categoriser.git
cd customer_categoriser

# 2. Environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# → fill in your MONGO_DB_URL, AWS keys

# 5. Run
uvicorn app:app --reload
```

API will be live at `http://localhost:8000`

---

## Roadmap

- [ ] Model confidence scores on predictions
- [ ] A/B comparison between model versions
- [ ] Evaluation metrics dashboard (live)
- [ ] MLflow experiment tracking
- [ ] CI/CD pipeline with automated tests
- [ ] Auto-retraining trigger on data drift
- [ ] Docker production optimisation
- [ ] Real-time prediction monitoring

---

## What This Demonstrates

| Capability                  | Implementation                                      |
|-----------------------------|-----------------------------------------------------|
| Full ML lifecycle           | Ingest → validate → transform → train → evaluate → push |
| Cloud integration           | MongoDB Atlas for data, AWS S3 for model storage    |
| MLOps thinking              | Timestamped artifacts, automated accept/reject gate |
| Backend deployment          | FastAPI + Render + GitHub auto-deploy               |
| Secret management           | `.env` pattern, excluded from Git history           |
| Production error handling   | Pipeline failures are caught and logged             |
| Model versioning            | S3 keys include timestamp; API loads latest         |

---

<div align="center">

**Built by [Shaswatik Giri](https://github.com/Shaswatik-gork) — Machine Learning Engineer**

*If this project helped you, consider giving it a ⭐*

</div>
