import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

"""
model : Araç modeli
year : Araç üretim yılı
* price : Araç fiyatı
transmission : Araç vites türü (Automatic,Manual, Semi-Auto)
mileage : Kat edilen mil sayısı
fuelType : Yakıt tipi (Petrol, Diesel)
tax : Yıllık vergi
mpg : Galon başına mil (galon, hacim ölçü birimi)
engineSize: Arabanın motor boyutu

Hedef : Arabalarının fiyatlarını tahmin eden bir regresyon modeli kurmak.
Not : Modelini MEAN SQUARE ERROR fonksiyonu ile test edilmelidir.


"""

#### Data Analysis
data = pd.read_csv("Machine Learning/Codes/ford_price_pred.csv")
print(data.head())
print(data.info())

# Are there any null values?
print(data.isnull().sum()) 
# Result : # We have no null value

sns.pairplot(data)  
plt.show() 

info= pd.DataFrame(data.isnull().sum(),columns=["IsNull"])
info.insert(1,"Duplicate",data.duplicated().sum(),True)
info.insert(2,"Unique",data.nunique(),True)
info.insert(3,"Min",data.min(),True)
info.insert(4,"Max",data.max(),True)
print(info)
# Result: There are 119 lines of data that repeat the same line and need to be removed
#         There are vehicles with 0 engine volume so they need to be examined

zero_engine_size = data[data["engineSize"] == 0.0]
print(zero_engine_size)
# Result: Since it is not possible for the engine volume to be 0, it must be removed




#### Categorical Data Analysis
def bar_plot(variable):
    global n
    var = data[variable] # get properties
    var_value = var.value_counts() # number of categorical variables
    
    plt.figure(figsize = (10,8)) # draw graph
    plt.bar(var_value.index, var_value)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.xticks(rotation=60)
    plt.show()
    
category = data.select_dtypes(include=['object']).columns
for i in category:
    bar_plot(i)

# Result: Fiesta and Focus vehicle brands have high sales
#         Manual vehicle more
#         Vehicles mostly use petrol and diesel fuel




#### Relationship between dependent variable
for  i in category:
    print(data[[i, "price"]].groupby([i]).mean().sort_values(by="price")[::-1])
    print("-"*25)

# Result: The most expensive vehicle is the mustang
#         Automatic transmission and hybrid vehicles are more expensive. In this case, categorical variables will be valuable to us.


#### Numerical Data Analysis
def plot_hist(variable):
    plt.figure(figsize=(7,3))
    plt.hist(data[variable], bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Graphic of {variable}")
    plt.show()

numeric = data.select_dtypes(include=['int64', "float64"]).columns
for i in numeric:
    plot_hist(i)
 



#### Correlation Analysis
sns.heatmap(data[numeric].corr(), annot=True, fmt=".2f")
plt.show()
# Result: The highest correlation is between year and price




#### Outlier Data Analysis
sns.boxplot(data = data.loc[:, numeric], orient = "v", palette = "Set1") 
plt.show()    
# Result: Let's also examine tax, mpg and enginesize. Scaling needed
sns.boxplot(data = data.loc[:, numeric[-3:]], orient = "v", palette = "Set1") 
plt.show()    
# Result: Outliers remove.

df = data.copy()

#### One Hot Encoder
data = pd.get_dummies(data, columns= ["transmission"], drop_first=True) # dummy value
data = pd.get_dummies(data, columns= ["fuelType"], drop_first=True) 

#### Label Encoder
label_encoder = LabelEncoder()
data["model"] = label_encoder.fit_transform(data["model"])
print(data.head())


# Local Outlier Factor(LOF) Method
X = data.drop(labels = "model", axis = 1).values
y = data["model"].values
clf = LocalOutlierFactor() 
y_pred = clf.fit_predict(X) 
print(y_pred) 
X_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame(-1*X_score, columns=["score"])
print(outlier_score.sort_values(by="score"))
# Result: It looks like an outlier above 3. Let's visualize it for the LOF method

threshold = 3
filtre = outlier_score["score"] > threshold
outlier_index = outlier_score[filtre].index.to_list()
x = pd.DataFrame(X, columns=data.columns[1:])
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="b", s=50) # outlier ones are colored blue
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="k", s=3)
radius = (X_score.max() - X_score) / (X_score.max() - X_score.min()) # normalization process
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], edgecolors="r", s=1000*radius, facecolors="none")
plt.show()


#### Base Model

X = data.drop(labels = "model", axis = 1)
y = data["model"]

# Let's split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Score :", round(r2_score(y_test, y_pred),2))
print("Mean_Squared_Error :", round(mean_squared_error(y_test, y_pred),2))



#### Data Regularization 

# Duplicate 
df = df.drop_duplicates() # rows removed
info= pd.DataFrame(df.isnull().sum(),columns=["IsNull"])
info.insert(1,"Duplicate",df.duplicated().sum(),True)
print(info)


# Engine capacity
df = df.drop(zero_engine_size.index)
zero_engine_size = df[df["engineSize"] == 0.0]
print(zero_engine_size) # removed


# Finding indices of outlier data
def detect_outlier(df_, features):
    outlier_indices = []
    for i in features:
        q1 = np.percentile(df_[i], 15)
        q3 = np.percentile(df_[i], 85)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step
        outlier_list = df_[(df_[i] < lower_bound) | (df_[i] > upper_bound)].index
        outlier_indices.extend(outlier_list)
    # If there is a lot of outlier data, instead of removing all of them, we can do it as follows
    print("Outliers : ", len(outlier_indices))
    outlier_indices = Counter(outlier_indices) # How many times do outlier data in the same column?
    print(outlier_indices) 
    print(outlier_indices.items())
    multiple_outlier = list(i for i, v in outlier_indices.items() if v >= 2) 
    print("Multi Outlier : ",len(multiple_outlier))
    return(multiple_outlier)

outliers = detect_outlier(df, numeric)


# Find outlier indices and replace them with the average
def replace_outliers_with_mean(df_, features):
    for i in features:
        q1 = np.percentile(df_[i], 15)
        q3 = np.percentile(df_[i], 85)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step
        outlier_indices = df_[(df_[i] < lower_bound) | (df_[i] > upper_bound)].index
        mean_value = df_[i].mean() 
        df_.loc[outlier_indices, i] = mean_value
    return(df_)

df = replace_outliers_with_mean(df, numeric[1:])

sns.boxplot(data = df.loc[:, numeric[:3]], orient = "v", palette = "Set1") 
plt.show()    
# Result: Let's also examine tax, mpg and enginesize. Scaling needed
sns.boxplot(data = df.loc[:, numeric[-3:]], orient = "v", palette = "Set1") 
plt.show()    



#### Encoding 
# Rare Encoder
def rare_analyser(df, cat_cols):
    for col in cat_cols:
        print(col, ":", len(df[col].value_counts()))
        print(pd.DataFrame({"COUNT": df[col].value_counts(),
                            "RATIO": df[col].value_counts()/len(df)}), end="\n\n\n")

rare_analyser(df, category[1:])
print("-"*50)

def rare_encoder(df, rare):
    df_copy = df.copy()
    rare_columns = [col for col in df_copy.columns if df_copy[col].dtype == "O" and (df_copy[col].value_counts() / len(df_copy) < rare).any()]
    for var in rare_columns:
        tmp = df_copy[var].value_counts() / len(df_copy)
        rare_label = tmp[tmp < rare].index
        df_copy[var] = np.where(df_copy[var].isin(rare_label), "Rare", df_copy[var])

    return(df_copy)

df = rare_encoder(df, 0.08)
rare_analyser(df , category[1:])


# One Hot Encoder
df = pd.get_dummies(df, columns= ["transmission"]) # dummy value
df = pd.get_dummies(df, columns= ["fuelType"], drop_first=True)
label_encoder = LabelEncoder()
df["model"] = label_encoder.fit_transform(df["model"])
print(df.head())



# Feature Enginering
# tax and engine size can be represented by a single variable
df["tax_engineSize"] = df["engineSize"] * df["tax"]


# Correlation
corr_matrix = df.corr()
print(corr_matrix["model"].sort_values(ascending=False)) 



#### Model Training
X = df.drop(labels = "model", axis = 1)
y = df["model"]

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Let's split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("RF_Score:", round(r2_score(y_test, y_pred), 2))
print("RF_Mean_Squared_Error:", round(mean_squared_error(y_test, y_pred), 2))


# Overfitting Check
print("RF_Train_Score : ", round(rf.score(X_train, y_train),2))
print("RF_Test_Score : ", round(rf.score(X_test, y_test),2))





xgb = XGBRegressor()
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
# Hyperparameter optimization with GridSearchCV
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=5, n_jobs=-1, return_train_score=True)
xgb_grid_search.fit(X_train, y_train)

y_pred = xgb_grid_search.predict(X_test)
print("XGB_Score:", round(r2_score(y_test, y_pred), 2))
print("XGB_Mean_Squared_Error:", round(mean_squared_error(y_test, y_pred), 2))


# Overfitting Check
print("XGB_Train_Score : ", round(xgb_grid_search.score(X_train, y_train),2))
print("XGB_Test_Score : ", round(xgb_grid_search.score(X_test, y_test),2))


# Result Analysis
results = xgb_grid_search.cv_results_

mean_train_scores = results['mean_train_score']
mean_test_scores = results['mean_test_score']
params = results['params']

best_index = xgb_grid_search.best_index_
best_train_score = mean_train_scores[best_index]
best_test_score = mean_test_scores[best_index]

plt.figure(figsize=(14, 7))
plt.plot(range(len(mean_train_scores)), mean_train_scores, label='Mean Train Score')
plt.plot(range(len(mean_test_scores)), mean_test_scores, label='Mean Test Score')
plt.scatter([best_index], [best_train_score], color='r', label='Best Train Score')
plt.scatter([best_index], [best_test_score], color='g', label='Best Test Score')
plt.xlabel('Parameter Combination Index')
plt.ylabel('R^2 Score')
plt.legend()
plt.title('Training and Test Scores for Different Parameter Combinations')
plt.show()
