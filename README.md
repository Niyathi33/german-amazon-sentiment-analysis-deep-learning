# German Amazon Review Sentiment Analysis using Deep Learning

## Overview

This project performs sentiment analysis on German Amazon reviews using both classical machine learning and deep learning models.

The goal is to compare different approaches for real-time sentiment prediction and evaluate their accuracy, performance, and latency.

Two pipelines are implemented:

1. TF-IDF + Logistic Regression (baseline)
2. Transformer-based model (XLM-RoBERTa)

The project demonstrates how deep learning models improve sentiment classification performance on multilingual text.



## Problem Statement

E-commerce platforms receive large numbers of user reviews every day.

Automatically detecting negative, neutral, and positive reviews helps:

- monitor customer satisfaction
- detect product issues early
- support real-time analytics dashboards

This project builds a sentiment classification system for German reviews and evaluates classical ML vs transformer models.



## Models Used

- TF-IDF + Logistic Regression (baseline)
- XLM-RoBERTa (Transformer)
- HuggingFace Transformers pipeline

Additional techniques:

- train / validation / test split
- confusion matrix evaluation
- precision / recall / F1-score
- latency measurement
- real-time inference helper



## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Datasets library
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SentencePiece
- Accelerate



## Files in Repository

- `Final_Realtime_sentiment_DE_EN_Niyathi.ipynb` → main notebook
- `final_realtime_sentiment_de_en_niyathi.py` → python script
- `DL_Final_Project_Report.docx` → final report
- `requirements.txt` → required libraries
- `data/` → dataset folder (not included)
- `results/` → output figures



## Dataset

The dataset used in this project is large and cannot be uploaded to GitHub.

To run the project:

1. Create a folder named `data`
2. Place dataset inside the folder
3. Update dataset path in the notebook

Example:

data/
   amazon_reviews.csv

Dataset source:
Amazon Reviews Multi (German)



## Results

Baseline (TF-IDF + Logistic Regression)

- Accuracy ≈ 0.60
- Macro-F1 ≈ 0.49

XLM-RoBERTa (Transformer)

- Accuracy ≈ 0.74
- Macro-F1 ≈ 0.66

Transformer model performs better, especially for neutral and negative reviews.



## Real-Time Inference

The notebook includes a helper function that allows prediction on new text:

Input → German review  
Output → sentiment + confidence  

This can be extended into:

- API
- dashboard
- real-time monitoring system



## Future Improvements

- train for more epochs
- use larger dataset
- deploy as web API
- build real-time dashboard



## Author

Niyathi Lekkala  
Master's in Computer Science  
