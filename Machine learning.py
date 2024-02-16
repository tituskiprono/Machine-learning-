import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler



df = pd.read_excel(r"D:\SORT\ml_project1_data.xlsx")
print(df)


#DATA CLEANING
df = pd.read_excel(r"D:\SORT\ml_project1_data.xlsx")
df.dropna(inplace = True)
print(df)

columns_to_replace = ["Income"]
df[columns_to_replace]=df[columns_to_replace].fillna("N/A")

df["Education"].unique()
df["Education"]= (df["Education"]=="Graduation").astype(int)
df["Marital_Status"].unique()
df["Marital_Status"] = (df["Marital_Status"]=="Married").astype(int)

remove = ("Dt_Customer")
df = df.drop(remove, axis=1)

plt.figure(figsize = (10,6))
plt.plot(df["Income"],df["MntWines"],label = "MntWines")
plt.plot(df["Income"],df["MntFruits"],label = "MntFruits")
plt.xlabel("Income")
plt.ylabel("MntWines,MntFruits")
plt.title("MACHINE PLOTS")
plt.legend()
plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
print("Train set:", len(train))
print("Validation set:", len(valid))
print("Test set:", len(test))

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
    
    
        
    data = np.hstack((x,np.reshape(y,(-1,1))))
    
    return data,x,y     
        
train,x_train,y_train = scale_dataset(train,oversample = True)
valid,x_valid,y_valid = scale_dataset(valid,oversample = False)
test,x_test,y_test = scale_dataset(test,oversample = False)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors = 100)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(x_train,y_train)

y_pred = nb_model.predict(x_test)
print(classification_report(y_test,y_pred))
        
    
    
    
    
    
    