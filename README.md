# regression-house-price

## 做法說明
A. 列出所有特徵值、找出每個特徵值有哪些類別，以及有沒有NAN值和缺失值

在這個步驟中，我們首先列出了資料集中的所有特徵值。這是為了讓我們了解我們有哪些特徵可以在後續的分析中使用。接著，我們使用自訂函式 feature_unfo 來找出每個特徵值的類別，並檢查是否存在缺失值或NAN值。這些資訊對於資料的初步探索和清理非常有幫助。

B. 將原始資料的"id"columns刪除

在這個步驟中，我們從原始資料集中刪除了名為 "id" 的欄位。這是因為 "id" 欄位通常是唯一識別每個樣本的值，對於建立模型和進行分析來說並不具有實際的預測能力，所以我們將其移除。

C. 離群值處理

在這個步驟中，我們針對特定特徵值（"condition", "floors", "yr_renovated", "waterfront", "view"）進行了離群值的處理。我們計算了這些特徵值的四分位數和IQR（四分位數範圍），然後使用1.5倍IQR的方法來識別和處理離群值。對於大於上四分位數的離群值，我們將其替換為上四分位數的值，對於小於下四分位數的離群值，我們將其替換為下四分位數的值。

D. 繪製每個特徵的箱型圖

在這個步驟中，我們繪製了每個特徵值的箱型圖，以可視化特徵值的分佈和檢查是否存在離群值。這些圖表有助於我們更好地理解資料的分佈特點，並幫助我們確認是否需要進一步處理離群值。

E. 將pricee改成正態分布，並計算改變前後的skewness and kurtosis偏度

在這個步驟中，我們針對目標變數 "price" 進行了數據轉換，將其轉換成正態分佈。我們使用Box-Cox轉換來實現這一目標，並計算轉換前後的偏度（skewness）和峰度（kurtosis）。這個步驟是為了使目標變數更符合線性模型的假設。

F. 使用XGBoost回歸模型，並用GridSearchCV搜索最佳參數

在這個步驟中，我們建立了一個XGBoost回歸模型，並使用GridSearchCV來搜索最佳的模型參數。我們定義了一個參數網格，包含各種參數的組合，然後使用交叉驗證來評估每個組合的性能。最終，我們找到了最佳參數組合並建立了一個最佳模型，並使用該模型對測試數據進行預測。

G. 將y_pred的值進行反轉換，並保存在excel文件

最後，在這個步驟中，我們對模型的預測結果進行了反轉換，以將其轉換回原始尺度。我們使用Box-Cox的反轉換公式來實現這一目標，然後將預測結果保存在Excel文件中，以便進一步分析或報告


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

### E.將pricee改成正態分布(設lmbda=0)，並計算改變前後的skewness and kurtosis偏度
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





