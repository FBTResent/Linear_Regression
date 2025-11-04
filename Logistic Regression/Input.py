import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
import Logistic_Regression as lgr

# Đọc dữ liệu và tạo nhãn
df = pd.read_csv("LG/WineQT.csv")
df["target"] = (df["quality"] >= 6).astype(int)
X = df.drop(["target", "quality", "Id"], axis=1).values
y = df["target"].values.reshape(-1,1)

# Chuẩn hóa và chia train/test
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Train Logistic Regression custom
model = lgr.LogisticRegression()
model.fit(X_train, y_train)
print(f"Epochs: {len(model.cost_history)}")

# So sánh với sklearn
from sklearn.linear_model import LogisticRegression as SklearnLR
sk_model = SklearnLR()
sk_model.fit(X_train, y_train.ravel())
y_pred_skl = sk_model.predict(X_test)
acc_skl = accuracy_score(y_test, y_pred_skl)
recall_skl = recall_score(y_test, y_pred_skl)
f1_skl = 2*acc_skl*recall_skl/(acc_skl+recall_skl)
print(f"Sklearn F1 score: {f1_skl:.4f}")

