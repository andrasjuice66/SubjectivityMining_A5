import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

# ... existing imports ...

def hard_majority_voting(models, X):
    """
    Perform hard majority voting ensemble.
    
    Args:
        models (list): List of trained models
        X (np.array): Input features
    
    Returns:
        np.array: Predicted labels
    """
    predictions = np.array([model.predict(X) for model in models])
    return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

def soft_majority_voting(models, X):
    """
    Perform soft majority voting ensemble.
    
    Args:
        models (list): List of trained models
        X (np.array): Input features
    
    Returns:
        np.array: Predicted labels
    """
    probabilities = np.array([model.predict_proba(X) for model in models])
    avg_probabilities = np.mean(probabilities, axis=0)
    return np.argmax(avg_probabilities, axis=1)

def stacking_ensemble(base_models, meta_model, X, y, n_splits=5):
    """
    Perform stacking ensemble with k-fold cross-validation.
    
    Args:
        base_models (list): List of base models
        meta_model: Meta-model for final prediction
        X (np.array): Input features
        y (np.array): True labels
        n_splits (int): Number of splits for k-fold cross-validation
    
    Returns:
        tuple: Trained meta-model and base models
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_features = np.zeros((X.shape[0], len(base_models)))
    
    for i, model in enumerate(base_models):
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train = y[train_index]
            
            model.fit(X_train, y_train)
            meta_features[val_index, i] = model.predict_proba(X_val)[:, 1]
    
    meta_model.fit(meta_features, y)
    return meta_model, base_models

# ... existing code ...

if __name__ == "__main__":
    # Load and preprocess data
    # ... (use your existing code to load and preprocess data) ...

    # Train individual models
    # ... (use your existing code to train individual models) ...

    # Assuming you have three trained models: model1, model2, model3
    models = [model1, model2, model3]

    # Hard Majority Voting
    hard_voting_predictions = hard_majority_voting(models, X_test)
    print("Hard Majority Voting Results:")
    print(classification_report(y_test, hard_voting_predictions))

    # Soft Majority Voting
    soft_voting_predictions = soft_majority_voting(models, X_test)
    print("Soft Majority Voting Results:")
    print(classification_report(y_test, soft_voting_predictions))

    # Stacking Ensemble
    meta_model = LogisticRegression()
    stacked_model, base_models = stacking_ensemble(models, meta_model, X_train, y_train)
    
    # Generate meta-features for test set
    test_meta_features = np.column_stack([model.predict_proba(X_test)[:, 1] for model in base_models])
    stacking_predictions = stacked_model.predict(test_meta_features)
    
    print("Stacking Ensemble Results:")
    print(classification_report(y_test, stacking_predictions))

# ... existing code ...