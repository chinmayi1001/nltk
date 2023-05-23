import nltk
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
text = "I will walk 500 miles and I would walk 500 more, just to be the man who walks a thousand miles to fall down at your door!"
tokens = nltk.word_tokenize(text)
print("Tokens:", tokens)

word_freq = Counter(tokens)
word_freq

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.casefold() not in stop_words]
filtered_tokens
nltk.download('averaged_perceptron_tagger')
pos_tags = nltk.pos_tag(filtered_tokens)
pos_tags