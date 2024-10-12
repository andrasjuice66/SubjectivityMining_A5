import re
import numpy as np
import pandas as pd
import contractions
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class Preprocessor:
    def __init__(self, max_words=20000, max_len=150):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.nlp = spacy.load('en_core_web_sm')
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def advanced_preprocess_text(self, text):
        text = self.expand_contractions(text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+|URL', '', text)
        text = re.sub(r'#', '', text)
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(tokens)

    def add_custom_features(self, data):
        data['text_length'] = data['text'].apply(len)
        data['special_char_count'] = data['text'].apply(lambda x: sum([1 for char in x if char in "!?."]))
        data['caps_count'] = data['text'].apply(lambda x: sum([1 for char in x if char.isupper()]))
        data['avg_word_length'] = data['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
        data['unique_words_ratio'] = data['text'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
        return data

    def preprocess_dataset(self, data, text_column='text', label_column='label', fit_tokenizer=True):
        """
        Preprocesses the entire dataset, including text preprocessing and feature engineering.

        Args:
            data (pd.DataFrame): Input DataFrame containing text and label columns.
            text_column (str): Name of the column containing the text data.
            label_column (str): Name of the column containing the labels.
            fit_tokenizer (bool): Whether to fit the tokenizer on the data.

        Returns:
            tuple: (X, y) where X is the preprocessed feature matrix and y is the label vector.
        """
        # Ensure the required columns exist
        assert text_column in data.columns, f"'{text_column}' column not found in the dataset"
        assert label_column in data.columns, f"'{label_column}' column not found in the dataset"

        # Preprocess text
        data['clean_text'] = data[text_column].apply(self.advanced_preprocess_text)

        # Add custom features
        data = self.add_custom_features(data)

        # Tokenize and pad sequences
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(data['clean_text'])
        
        sequences = self.tokenizer.texts_to_sequences(data['clean_text'])
        X = pad_sequences(sequences, maxlen=self.max_len)

        # Extract labels
        y = data[label_column].values

        return X, y

    def get_custom_features(self, X):
        return X[['text_length', 'special_char_count', 'caps_count', 'avg_word_length', 'unique_words_ratio']].values

    def transform_new_data(self, new_data, text_column='text'):
        """
        Transforms new data using the fitted preprocessor.

        Args:
            new_data (pd.DataFrame): New data to transform, containing a text column.
            text_column (str): Name of the column containing the text data.

        Returns:
            np.array: Transformed feature matrix for the new data.
        """
        assert text_column in new_data.columns, f"'{text_column}' column not found in the dataset"

        # Preprocess text
        new_data['clean_text'] = new_data[text_column].apply(self.advanced_preprocess_text)

        # Add custom features
        new_data = self.add_custom_features(new_data)

        # Tokenize and pad sequences
        sequences = self.tokenizer.texts_to_sequences(new_data['clean_text'])
        X = pad_sequences(sequences, maxlen=self.max_len)

        return X

    def get_vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def load_glove_embeddings(self, glove_file='glove.6B.100d.txt'):
        embeddings_index = {}
        with open(glove_file, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((self.max_words, 100))
        for word, i in self.tokenizer.word_index.items():
            if i >= self.max_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

# Example usage
if __name__ == "__main__":
    # Load your datasets
    olid_train = pd.read_csv('olid-train-small.csv')
    olid_test = pd.read_csv('olid-test.csv')
    hasoc_train = pd.read_csv('hasoc-train.csv')

    # Initialize the preprocessor
    preprocessor = Preprocessor()

    # Preprocess OLID-train-small dataset
    X_olid_train, y_olid_train = preprocessor.preprocess_dataset(olid_train, text_column='tweet', label_column='subtask_a')

    # Preprocess OLID-test dataset
    X_olid_test, y_olid_test = preprocessor.preprocess_dataset(olid_test, text_column='tweet', label_column='subtask_a', fit_tokenizer=False)

    # Preprocess HASOC-train dataset
    X_hasoc_train, y_hasoc_train = preprocessor.preprocess_dataset(hasoc_train, text_column='text', label_column='task_1', fit_tokenizer=False)

    embedding_matrix = preprocessor.load_glove_embeddings()


    print("OLID-train preprocessed shape:", X_olid_train.shape)
    print("OLID-test preprocessed shape:", X_olid_test.shape)
    print("HASOC-train preprocessed shape:", X_hasoc_train.shape)
    print("Vocabulary size:", preprocessor.get_vocab_size())
