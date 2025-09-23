# 🚀 Project Overview
This project is a **research study** on a **Hybrid Random Forest Classifier** that combines **ID3 Decision Trees** and **Support Vector Machines (SVM)** to leverage the strengths of both algorithms for improved classification accuracy and robustness. Unlike a standard Random Forest that relies solely on a single type of model, this implementation builds an ensemble of both handcrafted ID3 trees and powerful SVMs. This unique approach allows the model to capture complex relationships in data from different perspectives, significantly enhancing overall performance.

# ✨ Key Features
- **Hybrid Ensemble Learning**: This classifier is a hybrid ensemble model that combines ID3 trees and SVMs into a single, powerful system. The final prediction is determined by a majority vote among all models in the ensemble.

- **Custom ID3 Implementation**: Features a custom-built ID3 algorithm capable of handling both nominal and numeric data types by selecting the attribute and threshold that maximize information gain at each node.

- **Robust Testing**: The custom ID3 implementation was validated against scikit-learn's DecisionTreeClassifier and was found to achieve comparable performance across various datasets.

- **Comprehensive Evaluation**: The model's performance was thoroughly evaluated on four well-known datasets from Kaggle: Mushrooms, Wine Quality, Breast Cancer Wisconsin, and Crop Recommendation.

- **Detailed Experimental Plan**: The project includes a detailed plan to analyze the impact of different ID3/SVM ratios, the number of base models, and SVM parameters (parameter C) on classification quality, result stability, and execution time.

# 📈 Experimental Results & Conclusions

The project's findings reveal that there is no single best model, as the optimal configuration depends on the characteristics of the data, such as the number of classes and features.

- **Wine Quality Dataset**: Models based solely on ID3 performed best, with an average accuracy of 65.76% for 100 classifiers. This suggests that for small, multi-class datasets, simpler models may generalize better than more complex SVMs.

- **Breast Cancer Wisconsin Dataset**: All configurations achieved very high scores, with the best results (around 97-98% accuracy) obtained with a mix of ID3 and SVM models (ID3/SVM ratio of 0.75) and a high value for the C parameter. This indicates that the hybrid approach is effective for well-separated numerical data with a high number of features.

- **Crop Recommendation Dataset**: The hybrid configuration proved most effective. The best result was an accuracy of 98.77% with an ID3/SVM ratio of 0.5, 75 base models, and C=10. This demonstrates that combining different types of classifiers can better capture data relationships in complex, multi-class scenarios.

# 🛠️ Getting Started
The project is built with `Python 3.12` and requires several key dependencies, including `scikit-learn` for SVM models and comparison tests, and `pandas` and `numpy` for data manipulation. To get started, install the required packages using the `pyproject.toml` file.

# ✍️ Authors
- Natalia Pieczko
- Antoni Grajek
