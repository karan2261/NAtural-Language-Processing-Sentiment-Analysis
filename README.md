# Twitter Sentiment Analysis

## Overview
This project focuses on sentiment analysis of Twitter data, classifying tweets into four categories: Positive, Negative, Neutral, and Irrelevant. The implementation combines traditional machine learning techniques with anomaly detection using an autoencoder.

## Features
- **Data Preprocessing**: Text cleaning using SpaCy for better model performance.
- **Machine Learning Models**:
  - Random Forest Classifier
  - Naive Bayes Classifier
- **Autoencoder**: Used for anomaly detection based on reconstruction error.
- **Hyperparameter Tuning**: Performed using RandomizedSearchCV.
- **Data Visualization**: Includes word clouds for text analysis.

## Dataset
- **Training Data**: `twitter_training.csv`
- **Validation Data**: `twitter_validation.csv`

## Workflow
1. **Data Preprocessing**: Clean text by removing stop words, punctuation, and numbers using SpaCy.
2. **Feature Extraction**: Generate TF-IDF vectors from preprocessed text.
3. **Model Training**: Train Random Forest and Naive Bayes classifiers.
4. **Model Evaluation**: Evaluate using accuracy, precision, recall, and F1-score metrics.
5. **Hyperparameter Tuning**: Optimize model performance using cross-validation.
6. **Anomaly Detection**: Train an autoencoder to detect outliers in data.

## Results
- **Random Forest**:
  - Accuracy: 69.7%
  - Precision: 71.8%
  - Recall: 69.7%
  - F1-Score: 68.9%
- **Naive Bayes**:
  - Accuracy: 64.6%
  - Precision: 72.3%
  - Recall: 64.6%
  - F1-Score: 61.4%
- **Autoencoder**:
  - Average Reconstruction Error (Validation): 0.00176
  - Anomalies Detected: 3

## Key Takeaways
- **Text Preprocessing**: SpaCy improved text cleaning, enhancing the model performance.
- **Model Comparison**:
  - Random Forest performed better than Naive Bayes in accuracy and recall.
  - Naive Bayes was simpler and faster but less accurate.
- **Anomaly Detection**: The autoencoder shows promise for identifying anomalies, requiring further tuning.

## Future Work
- Explore deep learning techniques (e.g., Transformers) for sentiment classification.
- Experiment with other autoencoder architectures and thresholds for better anomaly detection.
