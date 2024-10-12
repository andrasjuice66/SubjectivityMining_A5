import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from preprocess import Preprocessor
from models import MLModel, CNNModel, BERTModel


class EnsembleMethods:
    def hard_majority_voting(models, X):
        predictions = np.array([model.predict(X) for model in models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def soft_majority_voting(models, X):
        probabilities = np.array([model.predict_proba(X) for model in models])
        avg_probabilities = np.mean(probabilities, axis=0)
        return np.argmax(avg_probabilities, axis=1)

    def stacking_ensemble(base_models, meta_model, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(base_models)))

        for i, model in enumerate(base_models):
            for train_index, val_index in kf.split(X):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold = y[train_index]

                model.fit(X_train_fold, y_train_fold)
                meta_features[val_index, i] = model.predict_proba(X_val_fold)[:, 1]

        meta_model.fit(meta_features, y)
        return meta_model, base_models

def setup_models(preprocessor, embedding_matrix):
    models = {
        'Logistic Regression': MLModel(model_type='log_reg'),
        'Random Forest': MLModel(model_type='random_forest'),
        'LightGBM': MLModel(model_type='lightgbm'),
        'CNN': CNNModel(preprocessor.get_vocab_size(), 100, 150, embedding_matrix),
        'BERT': BERTModel(model_type='bert', use_cuda=False),
        'hateBERT': BERTModel(model_type='hatebert', use_cuda=False)
    }
    return models

def train_and_evaluate(models, X_train, y_train, X_test, y_test, experiment_type):
    results = []
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if isinstance(model, BERTModel):
            # Prepare data for BERT models
            train_df = pd.DataFrame({'text': X_train.flatten(), 'labels': y_train})
            test_df = pd.DataFrame({'text': X_test.flatten(), 'labels': y_test})
            model.fit(train_df)
            predictions = model.predict(test_df['text'].tolist())
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        results.append({
            'Model': name,
            'Experiment': experiment_type,
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-score': report['weighted avg']['f1-score']
        })
        trained_models[name] = model
        print(f"Accuracy for {name}: {accuracy:.4f}")
    results_df = pd.DataFrame(results)
    return results_df, trained_models

def configure_and_run_ensemble(trained_models, ensemble_type, X, y):
    ensemble_methods = EnsembleMethods()
    models_list = list(trained_models.values())
    if ensemble_type == 'hard_voting':
        predictions = ensemble_methods.hard_majority_voting(models_list, X)
    elif ensemble_type == 'soft_voting':
        # Exclude models without predict_proba
        models_with_proba = [model for model in models_list if hasattr(model, 'predict_proba')]
        if not models_with_proba:
            print("No models with predict_proba available for soft voting.")
            return None, []
        predictions = ensemble_methods.soft_majority_voting(models_with_proba, X)
        models_list = models_with_proba
    elif ensemble_type == 'stacking':
        # Exclude models without predict_proba
        models_with_proba = [model for model in models_list if hasattr(model, 'predict_proba')]
        if not models_with_proba:
            print("No models with predict_proba available for stacking ensemble.")
            return None, []
        meta_model = LogisticRegression()
        stacked_model, base_models = ensemble_methods.stacking_ensemble(models_with_proba, meta_model, X, y)
        test_meta_features = np.column_stack([model.predict_proba(X)[:, 1] for model in base_models])
        predictions = stacked_model.predict(test_meta_features)
        models_list = base_models
    else:
        raise ValueError("Unsupported ensemble type")
    return predictions, models_list

def main():
    # Load datasets
    olid_train = pd.read_csv('olid-train-small.csv')
    olid_test = pd.read_csv('olid-test.csv')
    hasoc_train = pd.read_csv('hasoc-train.csv')

    # Initialize the preprocessor and preprocess datasets
    preprocessor = Preprocessor()
    X_olid_train, y_olid_train = preprocessor.preprocess_dataset(olid_train, text_column='tweet', label_column='subtask_a')
    X_olid_test, y_olid_test = preprocessor.preprocess_dataset(olid_test, text_column='tweet', label_column='subtask_a', fit_tokenizer=False)
    X_hasoc_train, y_hasoc_train = preprocessor.preprocess_dataset(hasoc_train, text_column='text', label_column='task_1', fit_tokenizer=False)
    embedding_matrix = preprocessor.load_glove_embeddings()

    # Set up models
    models = setup_models(preprocessor, embedding_matrix)

    # In-Domain Experiment
    print("\n================ In-Domain Experiment (Train on OLID, Test on OLID) ================")
    in_domain_results, trained_models_indomain = train_and_evaluate(models, X_olid_train, y_olid_train, X_olid_test, y_olid_test, experiment_type='In-Domain')

    # Cross-Domain Experiment
    print("\n================ Cross-Domain Experiment (Train on HASOC, Test on OLID) ================")
    cross_domain_results, trained_models_crossdomain = train_and_evaluate(models, X_hasoc_train, y_hasoc_train, X_olid_test, y_olid_test, experiment_type='Cross-Domain')

    # Combine and display individual model results
    all_results = pd.concat([in_domain_results, cross_domain_results], ignore_index=True)
    print("\n================ Individual Model Results ================")
    print(all_results[['Experiment', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-score']].to_string(index=False))

    # Ensemble configurations
    ensemble_results = []

    # In-Domain Ensembles
    print("\n================ In-Domain Ensemble Results ================")
    for ensemble_type in ['hard_voting', 'soft_voting', 'stacking']:
        # Define models to use
        if ensemble_type == 'hard_voting':
            ensemble_models = ['Logistic Regression', 'Random Forest', 'CNN', 'BERT', 'hateBERT']
        else:
            # For soft voting and stacking, only include models with predict_proba
            ensemble_models = [name for name in trained_models_indomain if hasattr(trained_models_indomain[name], 'predict_proba')]
            if not ensemble_models:
                print(f"No models with predict_proba for {ensemble_type} in In-Domain.")
                continue
        selected_models = {name: trained_models_indomain[name] for name in ensemble_models}
        predictions, used_models = configure_and_run_ensemble(selected_models, ensemble_type, X_olid_test, y_olid_test)
        if predictions is None:
            continue
        accuracy = accuracy_score(y_olid_test, predictions)
        report = classification_report(y_olid_test, predictions, output_dict=True, zero_division=0)
        ensemble_results.append({
            'Ensemble Type': ensemble_type.capitalize(),
            'Models Used': ', '.join(ensemble_models),
            'Experiment': 'In-Domain',
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-score': report['weighted avg']['f1-score']
        })
        print(f"\n{ensemble_type.capitalize()} Ensemble using models: {', '.join(ensemble_models)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_olid_test, predictions, zero_division=0))
        
    # Cross-Domain Ensembles
    print("\n================ Cross-Domain Ensemble Results ================")
    for ensemble_type in ['hard_voting', 'soft_voting', 'stacking']:
        # Define models to use
        if ensemble_type == 'hard_voting':
            ensemble_models = ['Logistic Regression', 'Random Forest', 'CNN', 'BERT', 'hateBERT']
        else:
            ensemble_models = [name for name in trained_models_crossdomain if hasattr(trained_models_crossdomain[name], 'predict_proba')]
            if not ensemble_models:
                print(f"No models with predict_proba for {ensemble_type} in Cross-Domain.")
                continue
        selected_models = {name: trained_models_crossdomain[name] for name in ensemble_models}
        predictions, used_models = configure_and_run_ensemble(selected_models, ensemble_type, X_olid_test, y_olid_test)
        if predictions is None:
            continue
        accuracy = accuracy_score(y_olid_test, predictions)
        report = classification_report(y_olid_test, predictions, output_dict=True, zero_division=0)
        ensemble_results.append({
            'Ensemble Type': ensemble_type.capitalize(),
            'Models Used': ', '.join(ensemble_models),
            'Experiment': 'Cross-Domain',
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-score': report['weighted avg']['f1-score']
        })
        print(f"\n{ensemble_type.capitalize()} Ensemble using models: {', '.join(ensemble_models)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_olid_test, predictions, zero_division=0))

    # Display ensemble results
    ensemble_results_df = pd.DataFrame(ensemble_results)
    print("\n================ Ensemble Model Results ================")
    print(ensemble_results_df[['Experiment', 'Ensemble Type', 'Models Used', 'Accuracy', 'Precision', 'Recall', 'F1-score']].to_string(index=False))

if __name__ == "__main__":
    main()
