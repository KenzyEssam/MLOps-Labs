import numpy as np
import litserve as ls
import pickle
import os
import pandas as pd
from src.deployment.online.requests import InferenceRequest


class InferenceAPI(ls.LitAPI):

    def setup(self, device="cpu"):
        model_path = "models/best_model.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])

            data = request["input"]

            x = pd.DataFrame([{
                "Pclass": data["Pclass"],
                "Sex": data["Sex"],
                "Age": data["Age"],
                "SibSp": data["SibSp"],
                "Parch": data["Parch"],
                "Fare": data["Fare"],
                "Embarked": data["Embarked"]
            }])

            return x

        except Exception as e:
            print("Decode error:", e)
            return None

    def predict(self, x):
        if x is None:
            return None

        return self.model.predict(x)

    def encode_response(self, output):
        if output is None:
            return {
                "message": "error",
                "prediction": None
            }

        return {
            "message": "success",
            "prediction": output.tolist()
        }