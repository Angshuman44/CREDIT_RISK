import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object




class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model= load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            print('Scaled Data')
            print((data_scaled))
            preds = model.predict(data_scaled)
        
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 gender: str,
                 car_owner: str,
                 property_owner: str,
                 children: int,
                 annual_income: float,
                 type_income: str,
                 education: str,
                 marital_status: str,
                 housing_type: str,
                 age_years: float,
                 years_employed: float,
                 family_members: int):

        self.gender = gender
        self.car_owner = car_owner
        self.property_owner = property_owner
        self.children = children
        self.annual_income = annual_income
        self.type_income = type_income
        self.education = education
        self.marital_status = marital_status
        self.housing_type = housing_type
        self.age_years = age_years
        self.years_employed = years_employed
        self.family_members = family_members

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "GENDER": [self.gender],
                "Car_Owner": [self.car_owner],
                "Propert_Owner": [self.property_owner],
                "CHILDREN": [self.children],
                "Annual_income": [self.annual_income],
                "Type_Income": [self.type_income],
                "EDUCATION": [self.education],
                "Marital_status": [self.marital_status],
                "Housing_type": [self.housing_type],
                "AGE_YEARS": [self.age_years],
                "YEARS_EMPLOYED": [self.years_employed],
                "Family_Members": [self.family_members],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)















































# class CustomData:
#     def __init__(  self,
#         gender: str,
#         race_ethnicity: str,
#         parental_level_of_education,
#         lunch: str,
#         test_preparation_course: str,
#         reading_score: int,
#         writing_score: int):

#         self.gender = gender

#         self.race_ethnicity = race_ethnicity

#         self.parental_level_of_education = parental_level_of_education

#         self.lunch = lunch

#         self.test_preparation_course = test_preparation_course

#         self.reading_score = reading_score

#         self.writing_score = writing_score

#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict = {
#                 "gender": [self.gender],
#                 "race_ethnicity": [self.race_ethnicity],
#                 "parental_level_of_education": [self.parental_level_of_education],
#                 "lunch": [self.lunch],
#                 "test_preparation_course": [self.test_preparation_course],
#                 "reading_score": [self.reading_score],
#                 "writing_score": [self.writing_score],
#             }

#             return pd.DataFrame(custom_data_input_dict)

#         except Exception as e:
#             raise CustomException(e, sys)