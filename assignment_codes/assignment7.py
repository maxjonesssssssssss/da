import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

text = "Machine learning allows computers to learn from data naturally"
print("Original:", text)

words = word_tokenize(text)
print("\n1. Tokens:", words)
print("\n2. POS Tags:", pos_tag(words))

filtered = [w for w in words if w not in stopwords.words('english') and w.isalpha()]
print("\n3. After Stop Words:", filtered)

stemmer = PorterStemmer()
print("\n4. Stems:", [stemmer.stem(w) for w in filtered])

lemmatizer = WordNetLemmatizer()
print("\n5. Lemmas:", [lemmatizer.lemmatize(w) for w in filtered])

docs = ["Machine learning is great",
        "Deep learning uses neural networks",
        "AI is the future of technology"]
tfidf  = TfidfVectorizer()
matrix = tfidf.fit_transform(docs)
df     = pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out())
print("\n6. TF-IDF Matrix:")
print(df.round(2))
