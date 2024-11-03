
import numpy as np
import pandas as pd
import torch

from model_trainer import ReservoirModel


class ReservoirPredictor:
    def __init__(self, model_path="models/reservoir_model.pth"):
        """Initialize the predictor with a trained model"""
        self.model = ReservoirModel()
        self.model.load_model(model_path)
        # No need to call eval() as it's done in load_model

        # Feature columns and scalers are loaded from the checkpoint
        self.feature_columns = self.model.feature_columns
        self.scaler_X = self.model.scaler_X
        self.scaler_y = self.model.scaler_y
        self.device = self.model.device

    def predict(self, input_data):
        """Make predictions for new data"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Ensure all required features are present
        missing_cols = set(self.feature_columns) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        # Get features in correct order
        X = input_data[self.feature_columns]

        # Scale input using the loaded scaler
        X_scaled = self.scaler_X.transform(X)

        # Make prediction
        self.model.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            prediction_scaled = self.model.model(X_tensor)
            prediction = self.scaler_y.inverse_transform(
                prediction_scaled.cpu().numpy()
            )

        # Add uncertainty bounds (example: ±15% for P10/P90)
        result = {
            "P50_rate": float(prediction[0][0]),
            "P10_rate": float(prediction[0][0] * 0.85),
            "P90_rate": float(prediction[0][0] * 1.15),
            "units": "STB/day",
        }

        return result


def generate_input_scenarios(n_scenarios=100):
    """Generate multiple realistic input scenarios"""
    param_ranges = {
        "days_on_production": (0, 365),
        "depth": (5000, 12000),
        "permeability": (1, 1000),
        "porosity": (0.05, 0.35),
        "initial_pressure": (2000, 6000),
        "temperature": (100, 250),
        "thickness": (20, 200),
        "initial_water_saturation": (0.1, 0.9),
        "water_cut": (0, 0.8),
        "flowing_pressure": (500, 4000),
    }

    scenarios = {}
    for param, (min_val, max_val) in param_ranges.items():
        if param == "permeability":
            scenarios[param] = np.exp(
                np.random.uniform(
                    low=np.log(min_val), high=np.log(max_val), size=n_scenarios
                )
            )
        elif param in ["porosity", "initial_water_saturation", "water_cut"]:
            a, b = 2, 2
            scenarios[param] = (
                np.random.beta(a, b, n_scenarios) * (max_val - min_val) + min_val
            )
        else:
            scenarios[param] = np.random.uniform(
                low=min_val, high=max_val, size=n_scenarios
            )

    return pd.DataFrame(scenarios)




def main():
    try:
        # Initialize predictor with explicit model path
        predictor = ReservoirPredictor("oil/models/reservoir_model.pth")

        # Generate scenarios
        n_scenarios = 100
        print("Generating input scenarios...")
        input_scenarios = generate_input_scenarios(n_scenarios)

        # Make predictions
        print("Making predictions...")
        predictions = []
        for _, scenario in input_scenarios.iterrows():
            pred = predictor.predict(scenario.to_dict())
            predictions.append(pred)

        # Create plots
        print("Creating visualizations...")

        # Save results
        print("Saving results...")
        results_df = input_scenarios.copy()
        results_df["P10_rate"] = [pred["P10_rate"] for pred in predictions]
        results_df["P50_rate"] = [pred["P50_rate"] for pred in predictions]
        results_df["P90_rate"] = [pred["P90_rate"] for pred in predictions]
        results_df.to_csv("./oil/results/prediction_results.csv", index=False)

        print("Analysis complete!")

    except FileNotFoundError:
        print(
            "Error: Model file not found. Please ensure the model file exists at the specified path."
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":
    main()
