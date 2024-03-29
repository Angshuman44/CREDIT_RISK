import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import read_sql_data
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split

from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self,data=None):
        try:

            if data is not None:
                #Reading from direct path
                df=pd.read_csv(os.path.join(data))
                logging.info("Reading completed from CSV/Excel file")


            else:
                ##reading the data from mysql
                df=read_sql_data()
                logging.info("Reading completed mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    traindata,testdata = obj.initiate_data_ingestion('notebook/Cleaned_and_processed.csv')
    objtransform= DataTransformation()
    df=pd.read_csv(traindata)
    print(df.columns)
    print(df.info())
    unique_values_dict = {}
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Iterate over each column and store unique values in a dictionary
    for column in categorical_columns:
        unique_values_dict[column] = df[column].unique()

    # Print the dictionary
    print(unique_values_dict)

    trainarr,testarr,_=objtransform.initiate_data_transormation(traindata,testdata)
    print(trainarr.shape)
    objmodel=ModelTrainer()
    print(objmodel.initiate_model_trainer(trainarr,testarr))