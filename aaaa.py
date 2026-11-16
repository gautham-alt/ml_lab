# ==========================================
# 1. BUILT-IN SCIKIT-LEARN DATASETS
# ==========================================
from sklearn.datasets import (
    load_iris,
    load_diabetes,
    load_digits,
    fetch_california_housing,
    fetch_20newsgroups,
    fetch_openml,
    load_files
)

# Small built-in datasets (as NumPy arrays)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

diabetes = load_diabetes()
X_diab, y_diab = diabetes.data, diabetes.target

digits = load_digits()
X_dig, y_dig = digits.data, digits.target

# Built-in dataset directly as a Pandas DataFrame
df_housing = fetch_california_housing(as_frame=True).frame

# Text dataset (20 Newsgroups)
news_train = fetch_20newsgroups(subset='train')
X_news, y_news = news_train.data, news_train.target

# OpenML dataset (Fashion MNIST)
X_fmnist, y_fmnist = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)


# ==========================================
# 2. TABULAR FILES (PANDAS)
# ==========================================
import pandas as pd

# Standard comma-separated CSV with header
df_csv = pd.read_csv('file.csv')

# Semicolon-separated CSV (e.g., Wine Quality)
df_wine = pd.read_csv('winequality-red.csv', sep=';')

# Tab-separated file
df_tab = pd.read_csv('diabetes.tab.txt', sep='\t')

# Whitespace-separated .data file without header & '?' as missing values (e.g., Auto MPG)
cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
df_mpg = pd.read_csv('auto-mpg.data', sep=r'\s+', names=cols, na_values='?')

# Compressed tabular files (.gz or single-file .zip)
df_gz = pd.read_csv('dataset.csv.gz')
df_zip = pd.read_csv('dataset.zip')

# Excel and JSON files
df_excel = pd.read_excel('dataset.xlsx')
df_json = pd.read_json('dataset.json')


# ==========================================
# 3. FOLDER OF TEXT DOCUMENTS (20 NEWSGROUPS LOCAL)
# ==========================================
# Requires folder structure: root_folder/class_name/text_files
local_news = load_files('20news-bydate-train', encoding='latin1')
X_local_news, y_local_news = local_news.data, local_news.target


# ==========================================
# 4. BINARY .UBYTE.GZ FILES (RAW FASHION MNIST)
# ==========================================
import gzip
import numpy as np

with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    y_train_idx = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    X_train_idx = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(y_train_idx), 784)


# ==========================================
# 5. KERAS / TENSORFLOW BUILT-IN DATASETS
# ==========================================
from tensorflow.keras.datasets import fashion_mnist

(X_tr_k, y_tr_k), (X_te_k, y_te_k) = fashion_mnist.load_data()
X_tr_k = X_tr_k.reshape(-1, 784) / 255.0
X_te_k = X_te_k.reshape(-1, 784) / 255.0


# ==========================================
# 6. PURE NUMPY NUMERIC MATRICES
# ==========================================
# Clean numeric CSV
raw_np = np.loadtxt('numeric_data.csv', delimiter=',')
X_np, y_np = raw_np[:, :-1], raw_np[:, -1]

# Numeric CSV containing missing values ('?')
raw_np_missing = np.genfromtxt('data_with_missing.csv', delimiter=',', missing_values='?', filling_values=np.nan)


# ==========================================
# 7. FOLDER OF RAW IMAGES (OPENCV)
# ==========================================
import os
import cv2

X_imgs, y_imgs = [], []
base_dir = 'image_dataset_folder'
for label, class_folder in enumerate(sorted(os.listdir(base_dir))):
    folder_path = os.path.join(base_dir, class_folder)
    for img_file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28)).flatten()
        X_imgs.append(img)
        y_imgs.append(label)

X_imgs, y_imgs = np.array(X_imgs), np.array(y_imgs)
