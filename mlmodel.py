'''
Simple linear regression model to predict stroke
Dataset: healthcare-dataset-stroke-data4.csv
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data4.csv")

# Chọn các cột số cần thiết
cdf = df[
    ['age', 'hypertension', 'heart_disease',
     'avg_glucose_level', 'bmi', 'stroke']
]

# Tách X và y
x = cdf.iloc[:, :-1]   # các đặc trưng
y = cdf.iloc[:, -1]    # nhãn stroke

# Khởi tạo mô hình
regressor = LinearRegression()

# Huấn luyện
regressor.fit(x, y)

# Lưu model
pickle.dump(regressor, open('model.pkl', 'wb'))

'''
# Load model và test thử
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[67, 1, 0, 228.69, 36.6]]))
'''
