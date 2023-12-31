# -*- coding: utf-8 -*-
"""2023ML-1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bl6yuQrNTZo8-Je1uzdn8apqZs_U8gBO
"""

from google.colab import drive
#我的雲端硬碟
drive.mount('/content/drive')

import pandas as pd
train= pd.read_csv('/content/drive/MyDrive/ML2023/hw1/train-v3.csv')
test= pd.read_csv('/content/drive/MyDrive/ML2023/hw1/test-v3.csv')
valid= pd.read_csv('/content/drive/MyDrive/ML2023/hw1/valid-v3.csv')

#列出所有特徵值
columns = train.columns.tolist
print(columns)

#找出每個特徵有哪些類別，以及NAN值有幾個
def feature_unfo(feature):
    print('有哪些類別' , feature.unique())
    print('有幾個NAN值:' , feature.isna().sum())
    print('是否存在缺失值' , feature.isnull().any())

#train.apply(feature_unfo , axis=0)

#print(train.shape)

#刪除沒有用的資料
train1 =train.drop(['id'],axis="columns")
valid1 =valid.drop(['id'],axis="columns")
test1 =test.drop(['id'],axis="columns")

#print(train1.shape)

#繪製箱型圖
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot(feature1):
    if feature1.name not in [ "condition" , "floors" , "yr_renovated" , "waterfront" , "view"]:
        print(feature1)
        print( '下四分位數Q1=' , feature1.quantile(0.25))
        print('中位數Q2=' ,feature1.quantile(0.5) )
        print('上四分位數Q3=' ,feature1.quantile(0.75) )
        print('IQR=' ,feature1.quantile(0.75)-feature1.quantile(0.25) )
        #upper boundary / lower boundary
        upper_boundary = feature1.quantile(0.75) + 1.5*(feature1.quantile(0.75)-feature1.quantile(0.25))
        lower_boundary = feature1.quantile(0.25) - 1.5*(feature1.quantile(0.75)-feature1.quantile(0.25))
        #離群值替換成upper boundary / lower boundary
        feature1 [feature1 < lower_boundary] = lower_boundary
        feature1 [feature1 > upper_boundary] = upper_boundary

selected_features = train1.drop([ "condition" , "floors" , "yr_renovated" , "waterfront" , "view"], axis=1)
selected_features.apply(boxplot, axis = 0)
train1[selected_features.columns] = selected_features #替換掉原本的, "condition" , "floors" , "yr_renovated" , "waterfront" , "view"
#print(train1.shape)

labels=train1.columns.tolist
for column in train1:
    print(train1[column].values)

# 分別繪製每一列的箱型圖


#plt.figure(figsize=(15,15))#設定畫布
num_rows = 1  # 每張圖的行數
num_cols = 2  # 每張圖的列數
num_subplots = 11  # 總共要畫的子圖數
# 遍歷生成每張圖片
for i in range(num_subplots):
    # 創建新的畫布
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 15))

    # 選擇要在這張圖片上顯示的箱型圖
    start_idx = i * num_cols
    end_idx = start_idx + num_cols
    selected_columns = train1.columns[start_idx:end_idx]

    # 遍歷每個子圖 (每張圖片上有 2 個子圖)
    for j, column in enumerate(selected_columns):
        ax = axes[j]
        sns.boxplot(data=train1[column], ax=ax)
        ax.set_title(f'箱型圖 {column}')
        ax.set_ylabel('值')

    # 調整子圖的排列
    plt.tight_layout()

    # 顯示畫布
    plt.show()

#改成正態分布
sns.distplot(train1.price)
plt.show()
#skewness and kurtosis偏度計算
print("Skewness: %f" % train1['price'].skew())
print("Kurtosis: %f" % train1['price'].kurt())
from scipy import stats

train1.price = stats.boxcox(train1.price,lmbda=0)
sns.distplot(train1.price)
plt.show()

#skewness and kurtosis偏度計算
print("Skewness: %f" % train1['price'].skew())
print("Kurtosis: %f" % train1['price'].kurt())



# 將 "price" 列作為目標變數 (y)
y_train = train1['price']
print(y_train.shape)
# 將其餘的列作為特徵 (X)
X_train = train1.drop(columns=['price'])  # 移除 "price" 列
print(X_train.shape)





# 將其餘的列作為特徵 (X)
X_test = test1
print(X_test.shape)

# 將 "price" 列作為目標變數 (y)
y_valid = valid1['price']
print(y_valid.shape)
# 將其餘的列作為特徵 (X)
X_valid = valid1.drop(columns=['price'])  # 移除 "price" 列
print(X_valid.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# 計算所有特徵與 'price' 之間的相關性
correlations = train1.drop("price", axis=1).corrwith(train1['price'])

# 創建一個熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heatmap')
plt.show()

import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

import tensorflow as tf

# 檢查GPU是否可用
if tf.test.gpu_device_name():
    print("GPU可用：", tf.test.gpu_device_name())
else:
    print("GPU不可用")

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 創建XGBoost回歸模型
xgb_model = xgb.XGBRegressor()

# 定義要搜索的參數網格

param_grid = {
    'min_child_weight': [1,3,5],
    'n_estimators': [500,700,900],
    'learning_rate': [0.01,0.03 ,0.05,0.1],
    'max_depth': [4,5,6],
    'gpu_id': [0],  # 指定使用的GPU設備
    'tree_method': ['gpu_hist'],  # 使用GPU訓練
}
# 标准化特征数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用GridSearchCV搜索最佳參數
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1)

# 開始搜索
grid_search.fit(X_train_scaled, y_train)

# 打印最佳參數和分數
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# 使用最佳參數的模型進行預測
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test_scaled)

import numpy as np

# 假设你的转换后的数据在 'train1.price' 中
lambda_value = 0  # 用于Box-Cox转换的lambda值，这里是0

# 执行反转换
original_values = np.exp(y_pred) - 1 #0

# original_values = ((y_pred * lambda_value) + 1)**(1 / lambda_value)

import pandas as pd

# 創建一個包含預測結果的DataFrame
result_df = pd.DataFrame({'Predicted Values': original_values})

# 將DataFrame保存為Excel文件
result_df.to_excel('/content/drive/MyDrive/ML2023/hw1/predicted_results.xlsx', index=False)