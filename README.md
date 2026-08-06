# WE Data & ML Internship Tasks

Internship project work for Telecom Egypt (WE), covering Python fundamentals and two end-to-end machine learning pipelines on telecom customer data.

## Contents

### 1. Python Assignment (`telecom_egypt_assignment.ipynb`)
Seven exercises covering Python fundamentals: input/output, conditionals, functions, dictionaries, file handling, and object-oriented programming (a subscriber management system with inheritance and exception handling).

### 2. Smart Plan Recommendation (`WE_ML_Smart_Plan_Recommendation.ipynb`)
An ML pipeline that recommends whether each subscriber should **upgrade**, **downgrade**, or **keep** their current plan. Joins five telecom tables, engineers behavioral features (tenure, average monthly revenue, add-on ratio, price-to-quota ratio), builds a rule-based label, and compares Logistic Regression, Random Forest, and Gradient Boosting. Ends with an estimated monthly revenue opportunity from upgrade candidates.

### 3. Ticket Triage — Classification, Sentiment & Summarization (`WE_ML_Complaint_Classification_Summarization.ipynb`)
A full NLP pipeline over customer support tickets, in 11 steps:
- **Synthetic data generation** — 500+ realistic tickets (Arabic / English / Franco-Arabic) generated with an LLM in batches, with programmatic deduplication and controlled noise injection.
- **EDA & cleaning** — Arabic normalization, noise removal, stopword filtering (negations preserved).
- **Feature engineering** — multilingual sentence embeddings, ticket length, exclamation count, lexicon sentiment score.
- **Modeling** — category classifier trained on embeddings; zero-shot pretrained Arabic sentiment model; extractive (TextRank) and abstractive (mT5) summarization.
- **Business impact** — urgent-ticket priority queue, estimated AHT reduction, top negative-sentiment categories with an operational recommendation.
- **Interactive demo** — a Gradio app that takes a raw ticket and returns category, sentiment, and summary.

## Tech Stack
Python, pandas, scikit-learn, sentence-transformers, Hugging Face Transformers, Gradio, matplotlib, seaborn.

## Note on Data
Datasets used are either provided for the internship or synthetically generated (Task 3). The synthetic tickets are **AI-generated, not real customer data**, produced purely to prototype the pipeline.

## Author
GitHub: [MarwanElGazzar](https://github.com/MarwanElGazzar)
