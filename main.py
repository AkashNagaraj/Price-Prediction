import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def pass_to_model(dataframe):
    
    return


def feature_selection(dataframe,num_columns,target_column):
    #1) Select the features which have atleast 30% correlation
    highly_correlated_features = dataframe.columns[~dataframe.corr()[target_column].between(-0.3,0.3, inclusive='both')]
    
    try:
    #2) Use information gain to filter out more features
        importances = mutual_info_classif(dataframe[highly_correlated_features.tolist()],dataframe[target_column])
        feat_importance = pd.Series(importances,highly_correlated_features).sort_values(ascending=False)

        # 3) Create a new dataframe with which includes only the most important features
        new_df = dataframe[feat_importance[1:num_columns+1].index.tolist()]
    except:
        print("Error extracting the features from the dataframe")

    return new_df


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


def read_data():
    dataframe = pd.read_csv("data/Ames_housing.csv")
    print("The shape of the data is",dataframe.shape)
    print("The head of the dataframe is",dataframe.head())
    # To get more information regarding the dataset perform dataframe.describe(), dataframe.info()
    return dataframe


def main():
    dataframe = read_data()
    target = "SalePrice"
    processed_dataframe = preprocess_data(dataframe)
    reduced_dataframe = feature_selection(processed_dataframe,10,target) 
    pass_to_model(reduced_dataframe)


if __name__=="__main__":
    main()
