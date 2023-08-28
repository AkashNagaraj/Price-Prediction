import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import argparse
import math
import sys

from simple_linear_regression import LinearRegression
from lasso_linear_regression import LassoRegression

def pass_to_model(dataframe,target):
    split_ratio = 0.8 
    model1 = LinearRegression(learning_rate=0.01, num_iterations=1000)
    scaler = StandardScaler()
    X = scaler.fit_transform(dataframe)
    y = (target-target.mean())/target.std()

    length = X.shape[0]
    split_length = math.ceil(length*split_ratio)
    X_train, X_test, y_train, y_test = X[:split_length], X[split_length:], y[:split_length], y[split_length:]
    
    model1.fit(X_train, y_train)
    predicted_values = model1.predict(X_test)
    mse = ((predicted_values - y_test.to_numpy())**2)
    print("MSE value is :", np.average(mse))

    model2 = LassoRegression(iterations = 1000, learning_rate = 0.01, l1_penality = 500)
    model2.fit(X_train, y_train)
    predicted_values = model2.predict(X_test)
    mse = ((predicted_values - y_test.to_numpy())**2)
    print("MSE value is :", np.average(mse))

    return


def feature_selection(dataframe,num_columns,target_column):
    try:
        #1) Select the features which have atleast 30% correlation
        highly_correlated_features = dataframe.columns[~dataframe.corr()[target_column].between(-0.3,0.3, inclusive='both')] 
        #print("The highly correlated features are :",highly_correlated_features)

        try:
        #2) Use information gain to filter out more features
            importances = mutual_info_classif(dataframe[highly_correlated_features.tolist()],dataframe[target_column])
            feat_importance = pd.Series(importances,highly_correlated_features).sort_values(ascending=False)

            # 3) Create a new dataframe with which includes only the most important features
            new_df = dataframe[feat_importance[1:num_columns+1].index.tolist()]
            important_features = feat_importance[1:num_columns+1].index.tolist()
        except:
            print("Error extracting the features from the dataframe")
            feat_importance = abs(dataframe.corr()["MEDV"]).sort_values(ascending=False)
            new_df = dataframe[feat_importance[:num_columns+1].index.tolist()]
            important_features = feat_importance[1:num_columns+1].index.tolist()
    except:
        important_features = abs(dataframe.corr()["SalePrice"]).sort_values(ascending=False).index.tolist()[1:11]
        new_df = dataframe[important_features]
    return new_df, important_features


def label_encoding(dataframe):
    l = LabelEncoder()
    for column in dataframe.columns:
        accepted_dtype = ["int64","float64","datetime64"]
        if dataframe[column].dtype not in accepted_dtype:
            dataframe[column] = l.fit_transform(dataframe[column])
    return dataframe


def remove_null_values(check_dataframe):
  if len(check_dataframe.columns[check_dataframe.isna().sum()>0]) > 0:
    #print("The columns with null values before preprocessing are : ",check_dataframe.columns[check_dataframe.isna().sum()>0])

    # 1) Remove the columns which have 40% or more null values
    if len(check_dataframe.columns[(check_dataframe.isna().sum()/check_dataframe.shape[0]) > 0.4].tolist())>0:
      #print("These columns have more than 40% null values hence they will be removed : \n",check_dataframe.columns[(check_dataframe.isna().sum()/check_dataframe.shape[0]) > 0.4].tolist())
      check_dataframe.drop(columns=check_dataframe.columns[(check_dataframe.isna().sum()/check_dataframe.shape[0]) > 0.4].tolist(), inplace=True)

    # 2) Fill in the Null values with the mean values
    check_dataframe.fillna(check_dataframe.mean(),inplace=True)
    #print("The columns with null values after forward fill are : \n",check_dataframe.columns[check_dataframe.isna().sum()>0])

    # 3) Fill in the mean for the remaining columns if type is int else remove them
    if len(check_dataframe.columns[check_dataframe.isna().sum()>0]) > 0:
        check_dataframe.dropna(inplace=True)

    if len(check_dataframe.columns[check_dataframe.isna().sum()>0]) > 0:
      return check_dataframe, False
    else:
      return check_dataframe,True
  else:
    return check_dataframe,True


def preprocess_data(dataframe):
    dataframe_without_null, check = remove_null_values(dataframe) 
    assert check==True, "Null values exist"
    processed_dataframe = label_encoding(dataframe_without_null)
    return processed_dataframe 


def read_data(dataname):
    if dataname=="Boston_housing":
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        dataframe = pd.read_csv(r"data/"+dataname+".csv", delim_whitespace=True, names=names)
    else:
        dataframe = pd.read_csv("data/"+dataname+".csv")
    # print("The shape of the data is",dataframe.shape, dataframe.head())
    # To get more information regarding the dataset perform dataframe.describe(), dataframe.info()
    return dataframe


def implement_online(dataframe,target):
    rows, col = dataframe.shape
    length = int(rows/100*70)
    
    scaler = StandardScaler()
    dataframe = scaler.fit_transform(dataframe)
    target = (target-target.mean())/target.std()

    # Fit 70% of the data
    X_initial, y_initial = dataframe[0:length], target[0:length]
    
    length = X_initial.shape[0]
    split_length = math.ceil(length*0.8)
    X_train, X_test, y_train, y_test = X_initial[:split_length], X_initial[split_length:], y_initial[:split_length], y_initial[split_length:]
    #X_train, X_test, y_train, y_test = train_test_split(X_initial,y_initial,test_size=0.2,random_state=5)
    reg = SGDRegressor(warm_start=True)
    reg.partial_fit(X_train,y_train)
    y_pred_reg = reg.predict(X_test)
    initial_error = ((y_pred_reg - y_test.to_numpy())**2)
    
    print(initial_error)
    sys.exit()

    # Online learning for 30%
    updated_errors = []
    for idx, rows in dataframe:
        reg.partial_fit([rows],[target[idx]])
        y_pred_reg = reg.predict(X_test)
        updated_errors.append(mean_squared_error(y_pred_reg,y_test))

    print(sum(updated_errors)/len(updated_errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data",help="Enter the name of the dataset")
    parser.add_argument("-t","--target",help="Enter the target column")
    args = parser.parse_args()

    dataframe = read_data(args.data)
    target = args.target
    processed_dataframe = preprocess_data(dataframe)
    reduced_dataframe, important_features = feature_selection(processed_dataframe,10,target) 
    #pass_to_model(reduced_dataframe, processed_dataframe[target])
    implement_online(reduced_dataframe[important_features], processed_dataframe[target])


if __name__=="__main__":
    main()
