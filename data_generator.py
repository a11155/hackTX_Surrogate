from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm


class SyntheticReservoirGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

        # Define realistic ranges for reservoir properties
        self.property_ranges = {
            "depth": (5000, 15000),  # ft
            "pressure": (2000, 8000),  # psi
            "temperature": (150, 300),  # °F
            "permeability": (1, 1000),  # mD
            "porosity": (0.05, 0.35),  # fraction
            "thickness": (20, 200),  # ft
            "initial_water_saturation": (0.15, 0.40),  # fraction
            "api_gravity": (20, 45),  # degrees
        }

    def generate_reservoir_properties(self, num_wells=20):
        """
        Generate base reservoir properties with spatial correlation
        """
        # Create a grid for spatial correlation
        grid_size = int(np.ceil(np.sqrt(num_wells)))
        x = np.linspace(0, 1000, grid_size)
        y = np.linspace(0, 1000, grid_size)
        X, Y = np.meshgrid(x, y)

        # Generate correlated properties using gaussian processes
        from scipy.stats import norm

        def correlated_property(mean, std, correlation_length=300):
            # Generate correlated field
            distances = np.sqrt(X**2 + Y**2)
            correlation = np.exp(-distances / correlation_length)

            # Generate random field
            z = np.random.normal(mean, std, size=X.shape)
            return z

        # Generate properties for each well
        wells = []
        well_locations = np.random.choice(
            len(x) * len(y), size=num_wells, replace=False
        )

        # Generate correlated properties
        depth_field = correlated_property(10000, 1000)
        perm_field = np.exp(correlated_property(np.log(100), 1))
        poro_field = correlated_property(0.2, 0.05)
        pressure_field = correlated_property(4000, 500)

        for i, loc in enumerate(well_locations):
            row, col = loc // grid_size, loc % grid_size

            well = {
                "well_id": f"W-{i+1:02d}",
                "x_coord": X[row, col],
                "y_coord": Y[row, col],
                "depth": depth_field[row, col],
                "permeability": perm_field[row, col],
                "porosity": np.clip(poro_field[row, col], 0.05, 0.35),
                "initial_pressure": pressure_field[row, col],
                "temperature": 75 + 0.02 * depth_field[row, col],  # Geothermal gradient
                "thickness": np.random.normal(100, 20),
                "initial_water_saturation": np.random.uniform(0.2, 0.35),
                "api_gravity": np.random.normal(35, 5),
            }

            wells.append(well)

        return pd.DataFrame(wells)

    def calculate_production_rate(self, t, initial_rate, di, b):
        """
        Calculate production rate using Arps' decline curve
        """
        if b == 0:  # Exponential decline
            return initial_rate * np.exp(-di * t)
        else:  # Hyperbolic decline
            return initial_rate / (1 + b * di * t) ** (1 / b)

    def calculate_initial_rate(self, well_properties):
        """
        Calculate initial production rate based on reservoir properties
        """
        # Productivity index calculation
        k = well_properties["permeability"]
        h = well_properties["thickness"]
        p = well_properties["initial_pressure"]

        # Basic PI calculation with realistic scaling
        pi = 0.00708 * k * h / (np.random.uniform(0.8, 1.2) * 20)  # Adding some noise

        # Initial rate calculation
        dp = p - 1000  # Assuming 1000 psi flowing bottom hole pressure
        initial_rate = pi * dp

        # Apply reasonable limits
        return np.clip(initial_rate, 100, 5000)

    def generate_production_data(self, well_properties, num_years=5):
        """
        Generate production data for each well
        """
        production_data = []
        start_date = datetime(2020, 1, 1)

        for _, well in well_properties.iterrows():
            # Calculate initial rate based on reservoir properties
            initial_rate = self.calculate_initial_rate(well)

            # Generate decline curve parameters
            di = np.random.uniform(0.2, 0.8)  # Initial decline rate
            b = np.random.uniform(0.3, 1.0)  # Hyperbolic exponent

            # Generate daily production data
            for day in range(num_years * 365):
                # Calculate basic production rate
                rate = self.calculate_production_rate(day / 365, initial_rate, di, b)

                # Add noise and periodic variations
                noise = np.random.normal(0, 0.05 * rate)
                seasonal_effect = 0.05 * rate * np.sin(2 * np.pi * day / 365)
                rate = max(0, rate + noise + seasonal_effect)

                # Calculate water cut (increasing with time)
                base_wc = well["initial_water_saturation"]
                water_cut = min(0.95, base_wc + 0.1 * (day / 365) ** 1.5)

                # Calculate GOR (increasing with pressure depletion)
                initial_gor = 1000  # scf/bbl
                gor = initial_gor * (1 + 0.1 * (day / 365))

                # Calculate flowing pressure (declining with time)
                p_flowing = well["initial_pressure"] * np.exp(-0.1 * day / 365)

                production_data.append(
                    {
                        "well_id": well["well_id"],
                        "date": start_date + timedelta(days=day),
                        "oil_rate": rate,
                        "water_cut": water_cut,
                        "gor": gor,
                        "flowing_pressure": p_flowing,
                        "days_on_production": day,
                    }
                )

        return pd.DataFrame(production_data)

   


def generate_dataset(num_wells=20, num_years=5, save_data=True):
    """
    Generate and optionally save synthetic reservoir dataset
    Returns well properties and production data DataFrames
    """
    generator = SyntheticReservoirGenerator()
    well_properties = generator.generate_reservoir_properties(num_wells)
    production_data = generator.generate_production_data(well_properties, num_years)

    if save_data:
        well_properties.to_csv("oil/data/well_properties.csv", index=False)
        production_data.to_csv("data/wroduction_data.csv", index=False)

    return well_properties, production_data



if __name__ == "__main__":
    well_properties, production_data = generate_dataset()
