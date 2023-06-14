import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self, 
    Gender: str,
    Branch: str,
    City: str,
    Customer_type: str,
    Product_line: str,
    Payment: str,
    Unit_price: float,
    Quantity: int,
    Tax: float,
    cogs: float,
    gross_margin_percentage: float,
    gross_income: float):
        self.Gender = Gender

        self.Branch = Branch

        self.City = City

        self.Customer_type = Customer_type

        self.Product_linee = Product_line

        self.Payment = Payment

        self.Unit_price = Unit_price

        self.Quantity = Quantity

        self.Tax = Tax

        self.cogs = cogs

        self.gross_margin_percentage = gross_margin_percentage

        self.gross_income = gross_income

def get_data_as_data_frame(self):
    try:
        custom_data_input_dict = {
                "Gender": [self.Gender],
                "Branch": [self.Branch],
                "City": [self.City],
                "Customer_type": [self.Customer_type],
                "Product_line": [self.Product_line],
                "Payment": [self.Payment],
                "Unit_price": [self.Unit_price],
                "Quantity": [self.Quantity],
                "Tax": [self.Tax],
                "cogs": [self.cogs],
                "gross_margin_percentage": [self.gross_margin_percentage],
                "gross_income": [self.gross_income],
        }

        return pd.DataFrame(custom_data_input_dict)

    except Exception as e:
        raise CustomException(e, sys)
