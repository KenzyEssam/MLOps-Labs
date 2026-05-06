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
            data = request["input"]

            # SINGLE RECORD
            if isinstance(data, dict):
                InferenceRequest(**data)

                x = pd.DataFrame([data])
                return x

            # BATCH REQUEST
            elif isinstance(data, list):
                for item in data:
                    InferenceRequest(**item)

                x = pd.DataFrame(data)
                return x

            else:
                raise ValueError("Invalid input format")

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