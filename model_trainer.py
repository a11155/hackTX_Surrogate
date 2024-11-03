import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class OilProductionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ReservoirNet(nn.Module):
    def __init__(self, input_dim):
        super(ReservoirNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


class ReservoirModel:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = None

    def prepare_data(self, well_properties, production_data):
        """
        Prepare and combine data for training
        """
        # Merge well properties with production data
        data = pd.merge(production_data, well_properties, on="well_id")

        # Select features
        self.feature_columns = [
            "days_on_production",
            "depth",
            "permeability",
            "porosity",
            "initial_pressure",
            "temperature",
            "thickness",
            "initial_water_saturation",
            "water_cut",
            "flowing_pressure",
        ]

        X = data[self.feature_columns]
        y = data["oil_rate"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, well_properties, production_data, epochs=100, batch_size=32):
        """
        Train the model
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            well_properties, production_data
        )

        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1))

        # Create datasets
        train_dataset = OilProductionDataset(X_train_scaled, y_train_scaled)
        test_dataset = OilProductionDataset(X_test_scaled, y_test_scaled)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model
        self.model = ReservoirNet(len(self.feature_columns)).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {train_loss/len(train_loader):.4f}")

        return self.evaluate(test_loader)

    def evaluate(self, test_loader):
        """
        Evaluate model performance
        """
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())

        predictions = self.scaler_y.inverse_transform(predictions)
        actuals = self.scaler_y.inverse_transform(actuals)

        from sklearn.metrics import mean_squared_error, r2_score

        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))

        return {
            "r2_score": r2,
            "rmse": rmse,
            "predictions": predictions,
            "actuals": actuals,
        }

    def save_model(self, path="models/reservoir_model.pth"):
        """
        Save trained model
        """
        if self.model is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y,
                    "feature_columns": self.feature_columns,
                },
                path,
            )

    def load_model(self, path="models/reservoir_model.pth"):
        """
        Load trained model
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.feature_columns = checkpoint["feature_columns"]
        self.model = ReservoirNet(len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler_X = checkpoint["scaler_X"]
        self.scaler_y = checkpoint["scaler_y"]
        self.model.eval()


if __name__ == "__main__":
    from data_generator import generate_dataset

    # Generate dataset
    well_properties, production_data = generate_dataset()

    # Train model
    model = ReservoirModel()
    results = model.train(well_properties, production_data)

    print("\nModel Performance:")
    print(f"R2 Score: {results['r2_score']:.3f}")
    print(f"RMSE: {results['rmse']:.2f}")

    # Save model
    model.save_model()
