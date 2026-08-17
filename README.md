# Wine Quality Classification Using SVM and XGBoost

This project implements and compares different machine learning classifiers (**Hard Margin SVM**, **Soft Margin SVM**, and **XGBoost**)
to predict wine quality using the real-world **Wine Quality dataset** (combining red and white wine data).

---

## 📌 Project Overview
The main objective of this assignment is to explore how different classification models handle real-world, complex, 
and imbalanced datasets, and how decision boundaries are formed across methods.

---

## 🔍 Data Preparation & Preprocessing
1. **Merging Datasets:** Combined red and white wine datasets and added a `type` feature to identify wine origin.
2. **Handling Duplicates:** Identified and removed 1,177 duplicate records to prevent training bias.
3. **Categorical & Label Encoding:** Converted the `type` feature into numerical values and applied Label Encoding to the target variable (`quality`).
4. **Feature Scaling:** Applied `StandardScaler` to normalize numerical features for optimal SVM performance.
5. **Class Imbalance Consideration:** Noted class imbalance (mostly medium quality scores 5 and 6); therefore, Accuracy, F1-score, and AUC were used for evaluation.

---

## 🤖 Models Description & Performance
* **Hard Margin SVM ($C = 1000$):** Achieved the lowest performance (Accuracy: 0.3581) because the dataset is not linearly separable.
* **Soft Margin SVM ($C = 1$):** Provided a flexible decision boundary, significantly improving performance and achieving the highest **AUC score (0.7808)**.
* **XGBoost Classifier:** Achieved the best overall **Accuracy (0.5536)** and **F1-score (0.5374)** by effectively capturing non-linear patterns and handling class imbalances.

---

## 🛠️ Tech Stack
* Python, Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn (`SVM`, `StandardScaler`, metrics)
* XGBoost

