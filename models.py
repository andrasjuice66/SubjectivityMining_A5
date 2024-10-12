import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import tensorflow as tf
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, Input, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

class MLModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='log_reg', **kwargs):
        self.model_type = model_type
        if model_type == 'log_reg':
            self.model = LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=50)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(num_leaves=50, learning_rate=0.05, n_estimators=100, 
                                            min_child_samples=30, subsample=0.8, colsample_bytree=0.8)
        else:
            raise ValueError("Unsupported model type")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    


class CNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, vocab_size, embedding_dim, max_len, embedding_matrix):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.max_len,))
        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix],
                              input_length=self.max_len, trainable=False)(inputs)

        convs = []
        filter_sizes = [3, 4, 5]
        for size in filter_sizes:
            conv = Conv1D(128, kernel_size=size, activation='relu')(embedding)
            pool = GlobalMaxPooling1D()(conv)
            convs.append(pool)

        concat = Concatenate()(convs)
        dropout = Dropout(0.5)(concat)
        outputs = Dense(1, activation='sigmoid')(dropout)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, validation_data, epochs=20, batch_size=64, class_weight=None, callbacks=None):
        return self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs, 
                              batch_size=batch_size, class_weight=class_weight, callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

class BERTModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='bert', use_cuda=True, num_labels=2):
        self.model_type = model_type
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        
        if model_type == 'bert':
            self.model = ClassificationModel('bert', 'bert-base-cased', use_cuda=use_cuda, num_labels=num_labels)
        elif model_type == 'hatebert':
            self.model = ClassificationModel('bert', 'GroNLP/hateBERT', use_cuda=use_cuda, num_labels=num_labels)
        else:
            raise ValueError("Unsupported model type. Choose 'bert' or 'hatebert'.")

    def fit(self, train_df, val_df=None, output_dir='output', preprocess_func=None):
        if preprocess_func:
            train_df['text'] = train_df['text'].apply(preprocess_func)
            if val_df is not None:
                val_df['text'] = val_df['text'].apply(preprocess_func)

        self.model.train_model(train_df, args={'output_dir': output_dir, 'overwrite_output_dir': True})
        
        if val_df is not None:
            result, model_outputs, wrong_predictions = self.model.eval_model(val_df)
            predictions, raw_outputs = self.model.predict(val_df['text'].tolist())
            return classification_report(val_df['labels'], predictions)
        
        return None

    def predict(self, X):
        predictions, _ = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        _, raw_outputs = self.model.predict(X)
        return raw_outputs
