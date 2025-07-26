import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGODB_URL")
print(MONGO_DB_URL)

import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

import certifi
ca=certifi.where()


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def csv_to_json_converter(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def insert_to_database(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL,tls=True,tlsCAFile=certifi.where(),serverSelectionTimeoutMS=10000)
            self.database=self.mongo_client[self.database]

            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        finally:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
        
if __name__=="__main__":
    try:
        FILE_PATH="Network_Data\phisingData.csv"
        DATABASE="SamyakAI"
        COLLECTION="NetworkData"
        networkobj=NetworkDataExtract()
        records=networkobj.csv_to_json_converter(FILE_PATH)
        print(records)
        no_of_records=networkobj.insert_to_database(records,DATABASE,COLLECTION)
        print(no_of_records)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
