# regression-house-price

## 做法說明



## 程式方塊圖與寫法

### A.列出所有特徵值、找出每個特徵值有哪些類別，以及有沒有NAN值和缺失值
```python
columns = train.columns.tolist
print(columns)
```
```python
def feature_unfo(feature):
    print('有哪些類別' , feature.unique())
    print('有幾個NAN值:' , feature.isna().sum())
    print('是否存在缺失值' , feature.isnull().any())
```
### B.將原始資料的"id"columns刪除
```python
train1 =train.drop(['id'],axis="columns")
valid1 =valid.drop(['id'],axis="columns")
test1 =test.drop(['id'],axis="columns")
```

### C.離群值處理
處理 "condition" , "floors" , "yr_renovated" , "waterfront" , "view"的離群值，將大於上四分位數的離群值替換成上四分位數；小於下四分位數的離群值替換成下四分位數

```python
def boxplot(feature1):
    if feature1.name not in ["condition" , "floors" , "yr_renovated" , "waterfront" , "view"]:
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

selected_features = train1.drop( "condition" , "floors" , "yr_renovated" , "waterfront" , "view"], axis=1)
selected_features.apply(boxplot, axis = 0)
train1[selected_features.columns] = selected_features #替換掉原本的"condition" , "floors" , "yr_renovated" , "waterfront" , "view"
```
### D.繪製每個特徵的箱型圖

```python
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
```
![01](diagrams/01.png)
![01](diagrams/02.png)
![01](diagrams/03.png)
![01](diagrams/04.png)
![01](diagrams/05.png)
![01](diagrams/06.png)
![01](diagrams/07.png)
![01](diagrams/08.png)
![01](diagrams/09.png)
![01](diagrams/10.png)
![01](diagrams/11.png)

### E.將pricee改成正態分布，並計算改變前後的skewness and kurtosis偏度
原本為右偏態，因此設lmbda=0效果最好

```python
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
```
![原始price分布](diagrams/12.png)
![轉換後price分布](diagrams/13.png)


### F.使用XGBoost回歸模型，並用GridSearchCV搜索最佳參數
```python
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
```
Best Parameters: {'gpu_id': 0, 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 900, 'tree_method': 'gpu_hist'}

### G.將y_pred的值進行反轉換，並保存在excel文件
```python

lambda_value = 0  
# 執行反轉換
original_values = np.exp(y_pred) - 1 

# 創建一個包含預測結果的DataFrame
result_df = pd.DataFrame({'Predicted Values': original_values})
# 將DataFrame保存為Excel文件
result_df.to_excel('/content/drive/MyDrive/ML2023/hw1/predicted_results.xlsx', index=False)
```



## 畫圖做結果分析
計算每個特徵值和price的相關度，並使用熱圖 (heatmap) 來視覺化各個特徵與目標變數之間的相關性

```python
# 計算所有特徵與 'price' 之間的相關性
correlations = train1.drop("price", axis=1).corrwith(train1['price'])

# 創建一個熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heatmap')
plt.show()
```

![01](diagrams/14.png)


## 討論預測值誤差很大的，是怎麼回事？
1. 模型的複雜度和特徵工程可能不足以捕捉到數據中的潛在變化和模式。這可能導致模型在預測時出現偏差
2. 數據質量可能是一個問題。低質量或不完整的數據集可能導致模型學習到不準確的模式，進而影響預測的準確性。
3. 如果數據中存在噪音，也可能導致預測誤差增大。
4. 模型的選擇和參數設定也可能影響預測的準確性。使用不適合特定任務的模型或不恰當的超參數設置可能導致誤差增大。


## 如何改進？
1.模型選擇





