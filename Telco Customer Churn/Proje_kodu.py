"""
Telekom şirketini terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

"""
CustomerId : Müşteri Id si
Gender : Cinsiyet
SeniorCitizen : Müşterinin yaşlı olup olmadığı (1,0)
Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
Dependents : Müşterinin bakmakla yükümlü olduğu kişilerin olup olmadığı (Evet, Hayır)(çocuk, anne, baba...)
tenure : Müşterinin şirkette kaldığı ay sayısı
PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity : Müşterinin çevrim içi güvenliğinin olup olmadığı (Evet, Hayır, Internet hizmeti yok)
OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, Internet hizmeti yok)
DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, Internet hizmeti yok)
TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, Internet hizmeti yok)
StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, Internet hizmeti yok)
StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, Internet hizmeti yok)
Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi, Kredi kartı)
MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
TotalCharges : Müşteriden tahsil edilen toplam tutar 
Churn : Müşterinin kullanıp kullanmadığı (Evet, Hayır)

"""

data = pd.read_csv("Telco-Customer-Churn.csv")

##### Recognizing Data
print(data.head())
print(data.shape)
print(data.info())

# Customer ID is unnecessary variable
data = data.drop(['customerID'], axis=1) 

# TotalCharges consists of numeric numbers, but the data type is object, let's convert it
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Let's make the target variable numerical
data["Churn"] = [1 if i=="Yes" else 0 for i in data["Churn"]]
print(data.info())

def check(df):
    print("  Shape  ".center(50, "#"))
    print(df.shape)
    print("\n")
    print("  Types  ".center(50, "#"))
    print(df.dtypes)
    print("\n")
    print("  Head  ".center(50, "#"))
    print(df.head())
    print("\n")
    print("  Tail  ".center(50, "#"))
    print(df.tail())
    print("\n")
    print("  Nan  ".center(50, "#"))
    print(df.isnull().sum())
    print("\n")
    print("  Quantiles  ".center(50, "#"))
    numeric_columns = df.select_dtypes(include=['int64', "float64"]) 
    print(numeric_columns.quantile([0, 0.05, 0.5, 0.95, 1]).T)
check(data)



##### Numerical and Categorical Variables Analysis
# Let's examine the data types
def analyze_columns(df, threshold1=10, threshold2=20):
    categoric_cols = [col for col in df.columns if df[col].dtype == "O"]
    numeric_categoric = [col for col in df.columns if df[col].nunique() < threshold1 and df[col].dtype != "O"]
    categoric_cardinal = [col for col in df.columns if df[col].nunique() > threshold2 and df[col].dtype == "O"]
    categoric_cols = categoric_cols + numeric_categoric
    categoric_cols = [col for col in categoric_cols if col not in categoric_cardinal]

    numeric_cols = [col for col in df.columns if df[col].dtype != "O"]
    numeric_cols = [col for col in numeric_cols if col not in numeric_categoric]

    print(f"Toplam Satır : {df.shape[0]}")
    print(f"Toplam Sütun : {df.shape[1]}")
    print(f"Toplam Kategorik Sütun : {len(categoric_cols)}")
    print(f"Toplam Numeric Sutun : {len(numeric_cols)}")
    print(f"Toplam Kategorik Kardinal Sütun: {len(categoric_cardinal)}")
    print(f"Toplam Nümerik Kategorik Sütun: {len(numeric_categoric)}")
    return(numeric_cols, categoric_cols)
numeric_cols, categoric_cols = analyze_columns(data)


# Relationship between variables and target variable
for col in data.columns[:-1]:
    print(f" {col} - Churn ".center(30, "#"))
    print(data[[col, "Churn"]].groupby([col]).mean().sort_values(by="Churn")[::-1].head())
    print("-"*50)

# Let's examine the tenure, MonthlyCharges, TotalCharges variables more meaningfully.
def target_summary(df, target, col_name):
    print(df.groupby(target).agg({col_name : "mean"}), end="\n\n\n")
for col in numeric_cols:
    target_summary(data, "Churn", col)






##### Missing Value Analysis
def missing_values(df):
    nan_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    num_miss = df[nan_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[nan_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([num_miss, np.round(ratio, 2)], axis=1, keys=["missing_number", "ratio"])
    print(missing_df, end="\n")
    print(f"\nTotal Missing Values : {num_miss.sum()}\n")
missing_values(data) 
#  There are 11 empty values ​​in TotalCharges. Let's fill it with median

data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Let's check for null value
print(data.isnull().sum())



##### Correlation Analysis
# Correlation check
sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f")
plt.show()
# Result: TotalCharges has high correlation with tenure
numeric_cols.append("Churn")
corr_matrix = data[numeric_cols].corr() # tek değişkenin diğerleri ile korelasyonu
print(corr_matrix["Churn"].sort_values(ascending=False)) 
numeric_cols.pop()
# Result: There is low correlation with the target variable


##### Base Model 
print(categoric_cols) 

# Since Churn and SeniorCitizain consist of 0 and 1 among categorical variables, they are included in numeric categorical variables, so we need to remove them.
categoric_cols = [i for i in categoric_cols if(i not in ['SeniorCitizen', 'Churn'])]
print(categoric_cols) 

# Encoding
def one_hot_encoder(df, categorical_cols, drop_first = False):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return(df)
colums1 = [col for col in categoric_cols if(10 >= data[col].nunique() > 2)]
colums2 = [col for col in categoric_cols if(col not in colums1)]
print(colums1)
print(colums2)

data = one_hot_encoder(data, colums1)
data = one_hot_encoder(data, colums2, True)
print(data.head()) 

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

model = CatBoostClassifier(verbose=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Başarı oranı : {round(accuracy_score(y_test, y_pred), 2)}")
# Result: Success 78%



##### Outlier Analysis
def outlier_tresholds(df, col_name, q1=0.05, q3=0.95):
    quartiel1 = df[col_name].quantile(q1)
    quartiel3 = df[col_name].quantile(q3)
    quartiel_range = quartiel3 - quartiel1
    up = quartiel3 + 1.5*quartiel_range
    down = quartiel3 - 1.5*quartiel_range
    return(up, down)

def check_outlier(df, col_name):
    up, down = outlier_tresholds(df, col_name)
    filter = df[(df[col_name] > up) | (df[col_name] < down)].any(axis=None)
    if(filter == True):
        return(True)
    else:
        return(False)

def replace_outliers(df, col_name):
    up, down = outlier_tresholds(df, col_name)
    df.loc[(df[col_name] < down), col_name] = down
    df.loc[(df[col_name] > up), col_name] = up

for col in numeric_cols:
    print(f"{col} : {check_outlier(data, col)}") 
    if(check_outlier(data, col)==True): 
        replace_outliers(data, col)
# Result: No outliers




##### Feature Enginering
data["New_tenure_Year"] = [
    "0-1" if (i >= 0 and i <= 12 ) else
    "1-2" if (i > 12 and i <= 24) else
    "2-3" if (i > 24 and i <= 36) else
    "3-4" if (i > 36 and i <= 48) else
    "4-5" if (i > 48 and i <= 60) else
    "5-6" for i in data["tenure"]
]

data["New_Engaged"] = [1 if(i == "One year" or i == "Two year") else 0 for i in data["Contract"]]
data["New_noProt"] = [1 if(i != "Yes" or j != "Yes" or k != "Yes") else 0 for i,j,k in zip(data["OnlineBackup"], data["DeviceProtection"], data["TechSupport"])]
data["New_Young_not_Engaged"] = [1 if(i == 0 and j == 0) else 0 for i,j in zip(data["New_Engaged"], data["SeniorCitizen"])]
data["New_Total_Services"] = (data[["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)
data["New_any_Streaming"] = [1 if(i == "Yes" or j == "Yes") else 0 for i,j in zip(data["StreamingTV"], data["StreamingMovies"])]
data["New_Otomatic_Pay"] = [1 if(i == "Bank transfer (automatic)" or i == "Credit card (automatic)") else 0 for i in data["PaymentMethod"]] 
data["New_Average_Charges"] = data["TotalCharges"] / (data["tenure"] + 1) 
data["New_Increase"] = data["New_Average_Charges"] / data["MonthlyCharges"] 
data["New_Services_Charges"] = data["MonthlyCharges"] / (data["New_Total_Services"] + 1) 

print(data.head())

##### Encoding

# Label Encoding
def label_encoder(df, binary_col):
    encoder = LabelEncoder()
    df[binary_col] = encoder.fit_transform( df[binary_col])
    return(df)
cols = [col for col in data.columns if(data[col].dtype not in [np.int64, np.float64] and data[col].nunique() == 2)]
print(cols)
# Here, it scans all the rows in the data and pulls only the columns that are not int and float but also have two different values.
print(data[cols].head())
for col in cols:
    label_encoder(data, col)
print(data[cols].head()) 


# One-Hot-Encoding
def one_hot_encoder(df, categorical_cols):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return(df)
colums = [col for col in data.columns if((data[col].dtype == object or data[col].dtype == str) and 10 >= data[col].nunique() > 2)]
print(colums)
# Here, it scans all the columns in the data and only pulls the columns with values ​​between 2 and 10.
print(data[colums].head())
data = one_hot_encoder(data, colums)
print(data.head()) 


##### Model Training

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(verbose=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Başarı oranı : {round(accuracy_score(y_pred, y_test), 2)}")
print("Test Scoru : ", model.score(X_test, y_test)) 
print("Train Scoru : ", model.score(X_train, y_train))
# Result: 80% success

def plot_importance(model, features, num):
    feature_imp = pd.DataFrame({"Value" : model.feature_importances_, 
                                "Features" : features.columns})
    plt.figure(figsize=(20,10))
    sns.barplot(data=feature_imp.sort_values(by="Value", ascending=False)[0:num], x="Value", y="Features")
    plt.show()
plot_importance(model, X, len(X))
# Result: New_increase shows that it is effective in increasing model success.  


