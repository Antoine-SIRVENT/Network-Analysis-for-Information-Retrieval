import nltk
import re  # Ajoutez cette ligne pour importer le module re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

def normalize_text(text, method='none'):
    """
    Normalise le texte par lemmatisation, stemmatisation ou ne fait aucun changement.

    Paramètres :
    - text (str) : Texte à normaliser.
    - method (str) : Méthode de normalisation ('lemma', 'stem' ou 'none').

    Retourne :
    - str : Texte normalisé ou inchangé.

    Lève :
    - ValueError : Si la méthode n'est ni 'lemma', ni 'stem', ni 'none'.
    """
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    if method == 'lemma':
        words = word_tokenize(text)
        processed_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(processed_words)
    elif method == 'stem':
        words = word_tokenize(text)
        processed_words = [stemmer.stem(word) for word in words]
        return ' '.join(processed_words)
    elif method == 'none':
        return text
    else:
        raise ValueError("Méthode invalide. Choisissez 'lemma', 'stem' ou 'none'.")

def clean_text(text, normalization_method="none"):
    """
    Nettoie le texte : conversion en minuscules, suppression des caractères non alphanumériques,
    puis normalisation selon la méthode choisie.
    """
    # Conversion en minuscules
    text = text.lower()
    # Suppression des caractères non alphanumériques (ponctuation, etc.)
    text = re.sub(r'\W+', ' ', text)
    # Normalisation (stemmatisation ou lemmatization)
    text = normalize_text(text, method=normalization_method)
    return text
