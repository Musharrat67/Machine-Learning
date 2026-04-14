# Question Pair Similarity Classification

# Project Overview
This project implements an AI solution to classify pairs of questions as "Duplicate" (semantically similar) or "Not Duplicate".

The core solution utilizes a Deep Learning approach (Siamese LSTM Network) with Transfer Learning (GloVe embeddings). To provide a performance benchmark, a Classical Machine Learning Baseline (SVM) was also implemented.

# Project Structure
* question-pair-similarity-classification-task.ipynb: The executable source code.
* model_performance_metrics.csv: Final comparison results.
* images: Directory containing ROC curves, Confusion Matrices, and EDA plots.
* train.csv: Dataset source.

# Setup & Run Instructions
1.  Environment: Python 3.x with TensorFlow, Scikit-Learn, Pandas, and NLTK.
2.  Data: Ensure `train.csv` and `glove.6B.100d.txt` are in the input directory.
3.  Execution: Run the Jupyter Notebook cells in order. The script automatically handles text preprocessing, model training on Dual GPUs, and evaluation.

---

# Approach & Key Decisions

# 1. Exploratory Data Analysis (EDA)
Initial analysis showed a class imbalance and variations in question length. Word clouds indicated significant vocabulary overlap in duplicates, but length analysis suggested that structure (word order) is critical, which motivated the use of sequence models (LSTM) over simple bag-of-words models.

# 2. Preprocessing Pipeline
NLP pipeline was applied:
* Cleaning: Lowercasing and regex removal of special characters.
* Normalization: Lemmatization to reduce vocabulary size (e.g., "running" -> "run").
* Vectorization: Used 100d GloVe embeddings for the LSTM (semantic) and TF-IDF for the SVM (statistical).

---

# Justification of Choices

# 1. Model Choice: Siamese LSTM
* Reasoning: The task requires comparing two distinct inputs to find a similarity score. A Siamese architecture is ideal because it processes both questions through identical sub-networks (shared weights), ensuring they are mapped to the same semantic vector space.
 Unlike Support Vector Machines (SVM) or Random Forests, LSTMs preserve the **sequential context** of words. For questions like "How do I..." vs "Do I know how...", the word order defines the meaning. Standard TF-IDF models miss this nuance.

# 2. Evaluation Metric: Recall & F1-Score
* Reasoning: Accuracy is misleading due to the class imbalance (more non-duplicates than duplicates).
* Focus on Recall: In a Q&A platform context, missing a duplicate (False Negative) forces users to re-answer the same question, which degrades platform quality. Therefore, I optimized the LSTM threshold to maximize Recall (81.21%), ensuring the model aggressively identifies potential duplicates.

# 3. Tuning Steps (Manual Optimization)
* Hyperparameter Search: Due to the strict 3-hour time constraint, I opted for manual iterative tuning rather than an exhaustive Grid Search, which would have exceeded the time limit on this dataset size.
* Key Tuning Decisions:
    * Batch Size (128): Selected to balance training speed on Dual T4 GPUs with sufficient gradient noise for generalization.
    * Learning Rate (0.0001): Lowered from standard 0.001 to prevent model collapse during training.
    * Threshold Tuning: Instead of a default 0.5 cutoff, I implemented an automated loop to find the optimal decision threshold (approx 0.33) that maximizes the F1-score.



# Performance Summary

| Model | Accuracy | Recall (Duplicates) | AUC-ROC |
| :--- | :--- | :--- | :--- |
| Siamese LSTM | 72.56% | 81.21% | 0.8178 |
| SVM Baseline | 77.88% | 70.77% | 0.8549 |

# Conclusion: While the SVM baseline achieved higher overall accuracy on clear-cut cases, the Siamese LSTM proved superior in Recall (+10.4%). This demonstrates that the Neural Network is more effective at the specific task of finding semantic duplicates, even if it trades off some precision to do so.
