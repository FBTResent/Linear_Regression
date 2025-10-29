import numpy as np
import pandas as pd
import Logistic_Regression as lgr

# Sinh dữ liệu
# np.random.seed(42)
# X = np.random.randint(0, 30, (200,13)) / 10
# z = -2*X[:,0] + -1*X[:,1] + 2*X[:,2]+2
# p = 1 / (1 + np.exp(-z))
# p[p>0.5]=1
# p[p<=0.5]=0
# # Huấn luyện
# lg = lgr.LogisticRegression()
# lg.fit(X, p, learning_rate=1e-2, epochs=1e+5,batches=10)
# print(lg.predict(X = np.random.randint(0, 30, (200, 13)) / 10))
# print("Best cost history:", lg.cost)
# print("Best parameter", lg.w)
# print(p)

# from sklearn.linear_model import LogisticRegression

# lg_skl = LogisticRegression()
# X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
# lg_skl.fit(X_b,p)
# epsilon = 1e-8
# m = len(p)
# y_pre = lg_skl.predict(X_b)

# from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score


# cm = confusion_matrix(y_pre, p)
# print(cm)
# [[94 1]
#  [2 103]]


# from sklearn.model_selection import train_test_split

# data = pd.read_csv('WineQT.csv')
# y = (data['quality'] >= 6).astype(int).values   # binary labels (0/1)
# X = data.drop(columns=['quality']).values      # feature matrix

# # then split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# lg = lgr.LogisticRegression()
# lg.fit(X_train, y_train, learning_rate=1e-2, epochs=10000,batches=10)
# y_pred = lg.predict(X_test)
# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# acc = accuracy_score(y_test, y_pred)
# print(cm)
# print("Accuracy:", acc)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

# Đọc dữ liệu
df = pd.read_csv("WineQT.csv")

# Tạo nhãn
df["target"] = np.where(df["quality"] >= 6, 1, 0)

# Chuẩn bị dữ liệu
X = df.drop(["target", "quality","Id"], axis=1)
y = df["target"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train Logistic Regression
model = lgr.LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán
threshold=0
f1=0
# Đánh giá
for threshold in np.linspace(0,1,100):
    y_pred = model.predict(X_test, threshold=threshold)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_ = 2*acc*recall/(acc+recall)
    if f1<f1_:
        f1=f1_
print(f"Threshold: {threshold}")
print("F1:", f1)


lgr_skl = LogisticRegression()
lgr_skl.fit(X_train,y_train)
y_pred_skl = lgr_skl.predict(X_test)
acc_skl = accuracy_score(y_test, y_pred_skl)
recall_skl = recall_score(y_test, y_pred_skl)
f1_skl = 2*acc_skl*recall_skl/(acc_skl+recall_skl)
print("Sklearn F1 score:",f1_skl)

