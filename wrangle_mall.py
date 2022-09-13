import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import acquire

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def wrangle_mall():
    '''
    custom function for wrangling mall from exercise guidance
    '''
    # get the data
    df = acquire.get_mallcustomer_data()

    #decide on target variable
    target = "spending_score"
    
    #return a summary with 2 dataframes
    cols_missing, rows_missing = summarize(df)

    #inital exploration of data
    univariate_explore(df)

    #return outliers and a list of columns that contain those values
    df_outlier,outlier_cols = get_outliers(df,k=1.5)

    #prep and clean the data, returning a modified dataframe
    mod_df = data_prep_drop(df, cols_to_remove=[], column_prop_required=.5, row_prop_required=.75)

    ## get dummies of any column that's an object
    for col in mod_df.select_dtypes(include=['object']).columns:
        mod_df = pd.concat([mod_df,pd.get_dummies(mod_df[col],drop_first=True)],axis=1)
        mod_df.drop(columns=col,inplace=True)
    
    ##split the data into three main sets, and seperate out the X and y variables
    X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test = split_tvt_continuous(mod_df,target)

    #Scale that data (all must be numerical)
    X_train_scaled, X_validate_scaled, X_test_scaled = \
        min_max_scale(X_train, X_validate, X_test, train.drop(columns=[target]).columns.tolist())

    ## good enough for this point
    return (df, 
            mod_df, 
            cols_missing, 
            rows_missing,
            df_outlier, 
            outlier_cols,
            X_train, y_train, 
            X_validate, 
            y_validate, 
            X_test, 
            y_test, 
            train, 
            validate, 
            test,
            X_train_scaled, 
            X_validate_scaled, 
            X_test_scaled
    )

def summarize(df):
    '''
    summary function that prints out the head, info, and describe of the passes dataframe as well as
    finds the nulls of col and nulls of row, attempts to categorize features as numerical or categorical
    if it is a categorical, it attemps to show value counts of each discrete, and binned version of numerical
    '''
    print('DataFrame head: \n')
    print(df.head())
    print('----------')
    print('DataFrame info: \n')
    print(df.info())
    print('----------')
    print('Dataframe Description: \n')
    print(df.describe())
    print('----------')
    print('Null value assessments: \n')
    cols_missing = nulls_by_col(df)
    print('nulls by column: ', cols_missing)
    print('--')
    rows_missing = nulls_by_row(df)
    print('nulls by row: ', rows_missing)
    numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
    categorical_cols = [col for col in df.columns if col not in numerical_cols ]
    print('--------')
    print('value_counts: \n')
    for col in df.columns:
        print('Column Name: ', col)
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
        print('--')
    print('---------')
    print('Report Finished')

    return cols_missing, rows_missing



def univariate_explore(df):
    ''' 
    takes in dataframe, and puts out a histogram of each category, binning relatively low
    '''
    plt.figure(figsize=(25, 5))
    for i, col in enumerate(df.columns.tolist()): # List of columns
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        plt.subplot(1,len(df.columns.tolist()), plot_number) # Create subplot.
        plt.title(col) # Title with column name.
        df[col].hist(bins=10) # Display histogram for column.
        plt.grid(False) # Hide gridlines.

    return

def nulls_by_col(df):
    ''' 
    takes in a data frame, finds the number and percentage of nulls by column, returns a dataframe of info
    '''
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
    {
        'num_rows_missing': num_missing,
        'percent_rows_missing': percnt_miss
    })
    return  cols_missing

def nulls_by_row(df):
    ''' 
    takes in a data frame, finds the number and percentage of nulls by row, returns a dataframe of info
    '''
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss})\
    .reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().reset_index().\
        rename(columns={"customer_id": 'count'})

    return rows_missing


def get_outliers(df, k):
   '''
    Given a series and a cutoff value, k (tukey value), returns the upper outliers for the series.
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound (q3 + (k*iqr)) the observation is.
   '''
   temp = df.copy()
   for col in temp.describe().columns:
      if not col.endswith('_outlier'):
         q1, q3 = temp[col].quantile([.25, .75])
         iqr = q3 - q1
         upper_bound = q3 + k * iqr
         lower_bound = q1 - k * iqr
         #print(col)
         temp[f"{col}_outlier"] = np.where((temp[col] - upper_bound) > 0,(temp[col] - upper_bound),
                                       np.where((temp[col] - lower_bound)<0,(temp[col] - lower_bound),0))

   outlier_cols = [col for col in temp if col.endswith('_outlier')]
   for col in outlier_cols:
      print('~~~\n' + col)
      data = temp[col][temp[col] > 0]
      print(data.describe())

   return temp,outlier_cols

def data_prep_drop(df, cols_to_remove=[], column_prop_required=.5, row_prop_required=.75):
    ''' 
    does the inital data_prep on a data set, this involves dropping any explictly stated columns,
    as well as dropping any columns or rows that do not meet the given thresholds (for non-null)
    '''
    df = df.drop(columns=cols_to_remove)
    threshold = int(round(column_prop_required*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(row_prop_required*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# Generic splitting function for continuous target.
def split_tvt_continuous(df,target):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    X_train, y_train, X_validate, y_validate, X_test, y_test = X_and_y(train,validate,test,target)

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test

def X_and_y(train,validate,test,target):
    '''
    takes in 4 variables (3 df and 1 string)
    just splits into X and y groups, nothing fancy
    returns 6 variables
    '''
        # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    takes in the train data sets (3) and numeric column list,
    and fits a min-max scaler to the first dataframe and transforms all 3
    returns 3 dataframes with the same scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled




def wrangle_example(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test= split_tvt_stratify(
        df, "target"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )
