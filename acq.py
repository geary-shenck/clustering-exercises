import pandas as pd
import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')