SVM — Binary Classification with 2 Features 🧪
This notebook compares Support Vector Machine (SVM) kernels—linear, rbf, poly, sigmoid—on a 2D subset of the Breast Cancer dataset to classify tumors as benign (0) or malignant (1).

Goal: Build a compact ML workflow (EDA → simple preprocessing → training → evaluation) and see how different SVM kernels behave in a low-dimensional setting.

🔹 Dataset & Features

Source columns: radius_mean, texture_mean, diagnosis

Label mapping: M → 1, B → 0

Train/Test split: 75% / 25%, random_state=15

🔹 What We Did

EDA & Prep: Quick structure check; select 2 features; map labels

Visualization: 2D scatter plot colored by class

Models: SVC with linear, rbf, poly, sigmoid

Metrics: classification_report + confusion_matrix

Tuning: GridSearchCV on RBF (C, gamma) to find best hyperparameters

🔹 Key Results (Test Set)

Linear SVM: ~0.85 accuracy (balanced precision/recall)

RBF SVM: ~0.87 accuracy (strong baseline)

Poly / Sigmoid: ~0.63 accuracy (collapsed to majority class)

Best RBF via GridSearch: C=10, gamma=0.01, ~0.86 accuracy with improved balance

🔹 Takeaways

With only two informative features, linear/RBF kernels perform reliably.

Poly/Sigmoid can underperform or collapse without careful tuning/feature scaling.

Simple 2D setup makes decision boundaries easy to interpret while demonstrating kernel effects.

🔹 Tech Stack

Python • pandas • numpy • scikit-learn • seaborn • matplotlib
