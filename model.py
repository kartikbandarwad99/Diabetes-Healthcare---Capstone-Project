import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv("diabetes.csv")

def imputation(df):
    # Columns with zeros in features except pregnancy make no sense. Hence they imply missing values
    cols_to_be_imputed=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for i in cols_to_be_imputed:
        df[i]=df[i].replace(0,df[i].mean())
        
    return df

def scaler(x):
    scale=StandardScaler()
    x=scale.fit_transform(x)
    pickle.dump(scale,open('scale.pkl','wb'))
    return x


df_imputed=imputation(df)    
x=df.drop(['Outcome'],axis=1)
x_scaled=scaler(x)
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y)
log_model=LogisticRegression()
log_model.fit(x_train,y_train)

pickle.dump(log_model,open('model.pkl','wb'))
        
preds=log_model.predict(x_test)

