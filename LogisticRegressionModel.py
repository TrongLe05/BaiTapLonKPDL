"""
Logistic Regression model to predict stroke
Dataset: healthcare-dataset-stroke-data4.csv
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data4.csv")

# Chọn các cột số cần thiết
cdf = df[
    ['age', 'hypertension', 'heart_disease',
     'avg_glucose_level', 'bmi', 'stroke']
]

# Tách X và y
x = cdf.iloc[:, :-1]   # đặc trưng
y = cdf.iloc[:, -1]    # nhãn (0 hoặc 1)

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression(
    max_iter=1000,     # tránh lỗi không hội tụ
    solver='lbfgs'
)

# Huấn luyện model
model.fit(x, y)

# Lưu model
pickle.dump(model, open('logistic_model.pkl', 'wb'))

"""
# Load model và test thử
model = pickle.load(open('logistic_model.pkl','rb'))

# Dự đoán xác suất bị stroke
proba = model.predict_proba([[67, 1, 0, 228.69, 36.6]])
print("Xác suất bị stroke:", proba[0][1])

# Dự đoán nhãn (0 hoặc 1)
prediction = model.predict([[67, 1, 0, 228.69, 36.6]])
print("Dự đoán:", prediction)
"""
