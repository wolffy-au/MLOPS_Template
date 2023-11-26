# 3. Model Evaluation and Testing Script (evaluate.py):
# This script assesses the performance of trained models using validation or test datasets.
# It may generate metrics, visualizations, or reports for model evaluation.
# Typically executed after model training to ensure model quality before deployment.

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

# # Evaluate predictions
# predictions = model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

evaluate_model(model, X_validation, Y_validation)