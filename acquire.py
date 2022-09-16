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

def get_mallcustomer_data():
    '''
    Reads in all fields from the customers table in the mall_customers schema from data.codeup.com
    
    parameters: None
    
    returns: a single Pandas DataFrame with the index set to the primary customer_id field
    '''
    if os.path.isfile("mall_customers.csv"):
        df = pd.read_csv("mall_customers.csv", index_col = 0)
    else:
        df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
        df.to_csv("mall_customers.csv")
    return df.set_index('customer_id')


def get_zillow_single_fam_2017_cluster():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_single_fam_sold_2017_cluster.csv"):
        df = pd.read_csv("zillow_single_fam_sold_2017_cluster.csv", index_col = 0)
    else:
        sql_query = """
                        SELECT  prop.*, 
                                pred.logerror, 
                                pred.transactiondate, 
                                air.airconditioningdesc, 
                                arch.architecturalstyledesc, 
                                build.buildingclassdesc, 
                                heat.heatingorsystemdesc, 
                                landuse.propertylandusedesc, 
                                story.storydesc, 
                                construct.typeconstructiondesc 
                        FROM    properties_2017 prop  
                        INNER JOIN (SELECT  parcelid,
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
                        -- WHERE  prop.latitude IS NOT NULL 
                            -- AND prop.longitude IS NOT NULL
                        WHERE propertylandusetypeid = 261
                            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                    """
        df = pd.read_sql(sql_query,get_connection("zillow"))
        df.to_csv("zillow_single_fam_sold_2017_cluster.csv")
    return df
