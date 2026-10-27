from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
train = fetch_20newsgroups(
    subset='train',
    remove=('headers', 'footers', 'quotes')
)

test = fetch_20newsgroups(
    subset='test',
    remove=('headers', 'footers', 'quotes')
)

# Convert text into word counts
vectorizer = CountVectorizer(
    stop_words='english',
    max_features=3000
)

X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

# MLE
mle = MultinomialNB(alpha=0)
mle.fit(X_train, y_train)

pred = mle.predict(X_test)

print("MLE Accuracy:", accuracy_score(y_test, pred))

# MAP with different Dirichlet priors
for alpha in [0.1, 1, 2, 10]:

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nAlpha:", alpha)
    print("Accuracy:", accuracy_score(y_test, pred))
