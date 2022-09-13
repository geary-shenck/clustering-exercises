import env
import pandas as pd
import os

def get_connection(db, user=env.user, host=env.host, password=env.password):
    ''' 
    basic synatx for getting the connection to connect to the database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_single_fam_2017():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_single_fam_sold_2017.csv"):
        df = pd.read_csv("zillow_single_fam_sold_2017.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt,fips,lotsizesquarefeet,regionidzip
                        FROM properties_2017 -- `2,858,627`, "2,985,217"
                            LEFT JOIN predictions_2017
                                USING (parcelid)
							WHERE propertylandusetypeid = 261
                            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                    ;
                    """
        df = pd.read_sql(sql_query,get_connection("zillow"))
        df.to_csv("zillow_single_fam_sold_2017.csv")
    return df
