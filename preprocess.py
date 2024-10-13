print("Starting script import...")

import re
print("re imported")
import numpy as np
print("numpy imported")
import pandas as pd
print("pandas imported")
import contractions
print("contractions imported")
from sklearn.feature_extraction.text import TfidfVectorizer
print("TfidfVectorizer imported")

import nltk
print("nltk imported")
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
print("nltk downloads completed")

import os
import urllib.request
import zipfile

class Preprocessor:
    def __init__(self, max_words=20000, max_len=150):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.max_words = max_words
        self.max_len = max_len
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        self.lemmatizer = WordNetLemmatizer()

    def expand_contractions(self, text):
        return contractions.fix(text)

    def advanced_preprocess_text(self, text):
        text = self.expand_contractions(text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+|URL', '', text)
        text = re.sub(r'#', '', text)
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(tokens)

    def add_custom_features(self, data):
        data['text_length'] = data['text'].apply(len)
        data['special_char_count'] = data['text'].apply(lambda x: sum([1 for char in x if char in "!?."]))
        data['caps_count'] = data['text'].apply(lambda x: sum([1 for char in x if char.isupper()]))
        data['avg_word_length'] = data['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
        data['unique_words_ratio'] = data['text'].apply(lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)
        return data

    def fit_on_texts(self, texts):
        for text in texts:
            for word in word_tokenize(text.lower()):
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
        
        sorted_words = sorted(self.word_counts, key=self.word_counts.get, reverse=True)
        for i, word in enumerate(sorted_words[:self.max_words - 1]):
            self.word_index[word] = i + 1
            self.index_word[i + 1] = word

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(word, 0) for word in word_tokenize(text.lower())] 
                for text in texts]

    def pad_sequences(self, sequences):
        return np.array([seq[:self.max_len] + [0] * max(0, self.max_len - len(seq)) 
                         for seq in sequences])

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
            self.fit_on_texts(data['clean_text'])
        
        sequences = self.texts_to_sequences(data['clean_text'])
        X = self.pad_sequences(sequences)

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
        sequences = self.texts_to_sequences(new_data['clean_text'])
        X = self.pad_sequences(sequences)

        return X

    def get_vocab_size(self):
        return len(self.word_index) + 1

    def load_glove_embeddings(self, glove_file='glove.6B.100d.txt'):
        glove_path = os.path.join('data', glove_file)
        
        # Check if the file already exists
        if not os.path.exists(glove_path):
            print(f"GloVe file not found. Downloading {glove_file}...")
            self.download_glove()
        
        print(f"Loading GloVe embeddings from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((self.max_words, 100))
        for word, i in self.word_index.items():
            if i >= self.max_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def download_glove(self):
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join('data', 'glove.6B.zip')
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Download the zip file
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        print("Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
        
        # Remove the zip file
        os.remove(zip_path)
        print("GloVe embeddings downloaded and extracted successfully.")

# Example usage
if __name__ == "__main__":
    print("Starting script...")
    
    # Load your datasets
    print("Loading datasets...")
    olid_train = pd.read_csv('data/olid-train-small.csv')
    olid_test = pd.read_csv('data/olid-test.csv')
    hasoc_train = pd.read_csv('data/hasoc-train.csv')
    print("Datasets loaded successfully.")

    # Initialize the preprocessor
    print("Initializing preprocessor...")
    preprocessor = Preprocessor()
    print("Preprocessor initialized.")

    # Preprocess OLID-train-small dataset
    X_olid_train, y_olid_train = preprocessor.preprocess_dataset(olid_train, text_column='text', label_column='labels')

    # Preprocess OLID-test dataset
    X_olid_test, y_olid_test = preprocessor.preprocess_dataset(olid_test, text_column='text', label_column='labels', fit_tokenizer=False)

    # Preprocess HASOC-train dataset
    X_hasoc_train, y_hasoc_train = preprocessor.preprocess_dataset(hasoc_train, text_column='text', label_column='labels', fit_tokenizer=False)

    embedding_matrix = preprocessor.load_glove_embeddings()


    print("OLID-train preprocessed shape:", X_olid_train.shape)
    print("OLID-test preprocessed shape:", X_olid_test.shape)
    print("HASOC-train preprocessed shape:", X_hasoc_train.shape)
    print("Vocabulary size:", preprocessor.get_vocab_size())
