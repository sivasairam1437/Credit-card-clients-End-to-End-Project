import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
            numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            ## Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            ##Rename column 'default payment next month' to Default
            train_df.rename(columns={"default payment next month": "Default"}, inplace=True)
            test_df.rename(columns={"default payment next month": "Default"}, inplace=True)

            #Grouping 0 category of marriage with 3 i.e; others
            train_df['MARRIAGE']=np.where(train_df['MARRIAGE'] == 0, 3, train_df['MARRIAGE'])
            test_df['MARRIAGE']=np.where(test_df['MARRIAGE'] == 0, 3, test_df['MARRIAGE'])



            #Grouping 0, 5, 6 categories in Education as 4 i.e; others
            train_df['EDUCATION']=np.where(train_df['EDUCATION'] == 5, 4, train_df['EDUCATION'])
            train_df['EDUCATION']=np.where(train_df['EDUCATION'] == 6, 4, train_df['EDUCATION'])
            train_df['EDUCATION']=np.where(train_df['EDUCATION'] == 0, 4, train_df['EDUCATION'])
            
            test_df['EDUCATION']=np.where(test_df['EDUCATION'] == 6, 4, test_df['EDUCATION'])
            test_df['EDUCATION']=np.where(test_df['EDUCATION'] == 5, 4, test_df['EDUCATION'])
            test_df['EDUCATION']=np.where(test_df['EDUCATION'] == 0, 4, test_df['EDUCATION'])

            #Changing the columns to categorical
            for att in ['SEX', 'EDUCATION', 'MARRIAGE', 'Default']:
                train_df[att] = train_df[att].astype('category')
                test_df[att] = test_df[att].astype('category')

            target_column_name = 'Default'
            drop_columns = [target_column_name,'ID']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Transformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            






        except Exception as e:
            logging.info("Exception occured in the initiate data transformation step")
            raise CustomException(e, sys)