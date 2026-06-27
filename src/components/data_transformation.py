import sys 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
import os 
from src.exception import CustomException
from src.utils import save_obj
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_features= ['writing_score', 'reading_score']
            cat_features= ['gender', 'race_ethnicity', 'lunch', 'test_preparation_course']
            ord_features=['parental_level_of_education']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ohe',OneHotEncoder(drop='first'))
                
                ]
            )
            ord_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(
                        categories=[[
                            'some high school',
                            'high school',
                            'some college',
                            "associate's degree",
                            "bachelor's degree",
                            "master's degree"]]))])
            logging.info("encoding all done")

            preprocessor=ColumnTransformer(
                transformers=[('num_pipeline',num_pipeline,num_features),
                              ('cat_pipeline',cat_pipeline,cat_features),
                              ('ord_pipeline',ord_pipeline,ord_features)],
                            remainder='passthrough'
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path) 
            test_df=pd.read_csv(test_path) 
            logging.info("Read train test data done")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name='math_score'

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing to train and test dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('saved preprocessing object')

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                    )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)