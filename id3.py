import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_excel(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
)

# 2. Preprocess
df = df.dropna(subset=['CustomerID'])

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Calculate total amount
df['Amount'] = df['Quantity'] * df['UnitPrice']

# 3. Customer-level features
customer = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'StockCode': 'nunique',
    'InvoiceDate': 'max',
    'Amount': 'sum'
}).reset_index()

customer.columns = [
    'CustomerID', 'Frequency', 'Quantity',
    'Products', 'LastPurchase', 'TotalAmount'
]

# Calculate recency
customer['LastPurchase'] = pd.to_datetime(customer['LastPurchase'])

customer['Recency'] = (
    customer['LastPurchase'].max() - customer['LastPurchase']
).dt.days

# 4. Create customer segments
customer['Segment'] = (
    customer['TotalAmount'] >
    customer['TotalAmount'].median()
).astype(int)

# Features and target
X = customer[
    ['Frequency', 'Quantity', 'Products', 'Recency']
]

y = customer['Segment']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 6. ID3-style Decision Tree
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Feature importance
print("\nFeature Importance:")

for name, importance in zip(X.columns, model.feature_importances_):
    print(name, ":", importance)

# 9. Visualize tree
plt.figure(figsize=(18, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Low Value', 'High Value'],
    filled=True
)

plt.show()
