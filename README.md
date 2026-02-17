🧠 Customer Categoriser
Production-Grade ML Segmentation System (MLOps + Cloud Deployment)


🌐 Live Application:
👉 https://customer-categoriser.onrender.com

🚀 What This Project Really Is
This is a fully automated, cloud-connected, versioned, deployable ML system that:

✔ Ingests production-style data from MongoDB
✔ Performs validation & drift detection
✔ Applies unsupervised clustering (KMeans)
✔ Enhances features using learned cluster intelligence
✔ Trains a supervised classifier (Logistic Regression)
✔ Evaluates performance with classification metrics
✔ Uploads trained models to AWS S3
✔ Automatically loads the latest versioned model
✔ Serves predictions via FastAPI
✔ Deploys automatically via Render

This project demonstrates real-world MLOps architecture, not just model training.

🏗 System Architecture
              src/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
│
├── pipeline/
│   ├── train_pipeline.py
│   └── prediction_pipeline.py
│
├── configuration/
├── entity/
├── cloud_storage/
└── artifact/

🧠 Machine Learning Strategy
1️⃣ Hybrid Intelligence Approach

Instead of directly classifying customers, the system:

Step 1 — Unsupervised Learning

KMeans Clustering

Discovers natural customer segments

Adds cluster label as feature

Step 2 — Supervised Learning

Logistic Regression

Learns decision boundaries enhanced by cluster intelligence

Outputs final customer category

This hybrid approach improves segmentation intelligence beyond basic classification.

📊 Model Evaluation

Metrics used:

F1 Score

Precision

Recall

Automated acceptance check before deployment

Models that fail evaluation do not get pushed forward.

⚙️ Tech Stack (Production Focused)
🖥 Backend

FastAPI

Uvicorn

Jinja2 Templates

CORS Middleware

🧮 Machine Learning

scikit-learn

KMeans

Logistic Regression

NumPy

Pandas

🗄 Database

MongoDB Atlas

☁️ Cloud & Storage

AWS S3 (Model Storage)

Automatic model version tracking

🚀 Deployment

Render (Free Tier)

GitHub Auto Deploy

📂 Clean Modular Architecture
src/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
│
├── pipeline/
│   ├── train_pipeline.py
│   └── prediction_pipeline.py
│
├── configuration/
├── entity/
├── cloud_storage/
└── artifact/


This structure mirrors industry-level ML system design.

🔁 Automated Model Lifecycle

Every time training runs:

New artifact folder is created (timestamped)

Model is trained and evaluated

If accepted:

Stored locally

Uploaded to AWS S3

FastAPI automatically loads latest trained model

No manual intervention required.

🔐 Secure Environment Design

All secrets handled via environment variables:

MONGO_DB_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION


Secrets are excluded from Git history.

🧪 Run Locally

Clone:

git clone https://github.com/Shaswatik-gork/-Project-Customer_Categoriser.git
cd customer_categoriser


Create virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run:

uvicorn app:app --reload

🎯 What This Project Demonstrates

This repository shows capability in:

✔ Full ML lifecycle automation
✔ Real cloud integration (AWS + MongoDB)
✔ MLOps thinking
✔ Backend deployment architecture
✔ Secret management & Git hygiene
✔ Debugging complex pipelines
✔ Production error handling
✔ FastAPI integration with trained models

This is not a toy ML demo.
This is portfolio-level ML engineering work.

📈 Future Enhancements

Add model confidence scores

Implement A/B model comparison

Add dashboard for evaluation metrics

Integrate MLflow

CI/CD pipeline testing

Auto retraining trigger

Docker production optimization

Real-time monitoring

👨‍💻 Author

Shaswatik Giri
Machine Learning Engineer 
