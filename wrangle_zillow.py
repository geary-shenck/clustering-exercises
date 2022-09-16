import pandas as pd
import numpy as np
import os
import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## begin acquire step
def get_zillow_single_unit_2017_cluster():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_single_fam_sold_2017_cluster.csv"):
        df = pd.read_csv("zillow_single_fam_sold_2017_cluster.csv", index_col = 0)
    else:
        sql_query = """
                        SELECT  properties_2017.*, 
                                pred.logerror, 
                                pred.transactiondate, 
                                air.airconditioningdesc, 
                                arch.architecturalstyledesc, 
                                build.buildingclassdesc, 
                                heat.heatingorsystemdesc, 
                                landuse.propertylandusedesc, 
                                story.storydesc, 
                                construct.typeconstructiondesc 
                        FROM    properties_2017 
                        JOIN (SELECT    parcelid,
                                        logerror,
                                        Max(transactiondate) transactiondate 
                                    FROM   predictions_2017 
                                    GROUP  BY parcelid, logerror) pred
                        USING (parcelid) 
                        LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
                        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
                        LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
                        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
                        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
                        LEFT JOIN storytype story USING (storytypeid) 
                        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
                         WHERE  properties_2017.latitude IS NOT NULL 
                            AND properties_2017.longitude IS NOT NULL
                            -- AND propertylandusetypeid = 261
                            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                    """
        df = pd.read_sql(sql_query,get_connection("zillow"))
        df.to_csv("zillow_single_fam_sold_2017_cluster.csv")
    return df

def get_connection(db, user=env.user, host=env.host, password=env.password):
    ''' 
    basic synatx for getting the connection to connect to the database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# end acquire step


# begin prepare and tidy step
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
    #for col in df.columns:
    #    print('Column Name: ', col)
    #    if col in categorical_cols:
    #        print(df[col].value_counts())
    #    else:
    #        print(df[col].value_counts(bins=10, sort=False))
    #    print('--')
    print('---------')
    print('Report Finished')

    return cols_missing, rows_missing

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
        rename(columns={"index": 'count'})

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
         upper_bound = q3 + (k * iqr)
         lower_bound = q1 - (k * iqr)
         #print(col)
         temp[f"{col}_outlier"] = np.where((temp[col] - upper_bound) > 0,(temp[col] - upper_bound),
                                       np.where((lower_bound - temp[col])>0,(lower_bound - temp[col]),0))

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



def get_numeric_X_cols(X_train, object_cols):
    """
    RUN THIS AFTER OBJECT COLUMNS
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def split_tvt_continuous(df,target):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test


def min_max_scale(X_train, X_validate, X_test, scale_cols):
    """
    takes in the train data sets (3) and numeric column list,
    and fits a min-max scaler to the first dataframe and transforms all 3
    returns 3 dataframes with the same scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[scale_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[scale_cols])
    X_validate_scaled_array = scaler.transform(X_validate[scale_cols])
    X_test_scaled_array = scaler.transform(X_test[scale_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=scale_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=scale_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=scale_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled
