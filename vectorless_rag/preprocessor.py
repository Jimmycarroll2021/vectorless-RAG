"""
Text preprocessing module for the vectorless RAG system.
"""

import re
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class Preprocessor:
    """Handles text preprocessing for documents and queries."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
        min_token_length: int = 2
    ):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_stopwords: Remove common stopwords
            remove_punctuation: Remove punctuation marks
            stemming: Apply stemming (Porter stemmer)
            lemmatization: Apply lemmatization
            min_token_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_token_length = min_token_length

        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if stemming else None
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None

    def preprocess(self, text: str) -> str:
        """
        Preprocess a text string.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue

            # Skip stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Skip pure numbers (optional)
            if token.isdigit():
                continue

            # Apply stemming
            if self.stemming and self.stemmer:
                token = self.stemmer.stem(token)

            # Apply lemmatization
            if self.lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)

            processed_tokens.append(token)

        return ' '.join(processed_tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text into tokens.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        processed_text = self.preprocess(text)
        return processed_text.split()
