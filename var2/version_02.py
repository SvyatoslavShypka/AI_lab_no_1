import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def linear_model(x, a, b):
    return a * x + b

def analyze_data(file_path):
    # Wczytanie danych
    # data = pd.read_csv(file_path)
    data = pd.read_csv(file_path,
                     dtype={"id": int, "company": str, "line": str, "departure_time": str, "arrival_time": str
                         , "start_stop": str, "end_stop": str, "start_stop_lat": float, "start_stop_lon": float,
                            "end_stop_lat": float, "end_stop_lon": float})
    data.rename(columns={"start_stop_lat": "x", "start_stop_lon": "y"}, inplace=True)
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("Plik CSV musi zawierać kolumny 'x' i 'y'")

    x_data = data['x'].values
    y_data = data['y'].values

    # Dopasowanie modelu liniowego
    params, covariance = curve_fit(linear_model, x_data, y_data)
    a, b = params

    # Obliczenie wartości dopasowania
    y_fit = linear_model(x_data, a, b)

    # Obliczenie współczynnika determinacji R^2
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Wizualizacja danych
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label='Dane', color='blue')
    plt.plot(x_data, y_fit, label=f'Dopasowanie: y = {a:.2f}x + {b:.2f}\n$R^2$ = {r_squared:.3f}', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Analiza danych')
    plt.grid(True)
    plt.show()

    return {'a': a, 'b': b, 'R^2': r_squared}

if __name__ == "__main__":
    file_path = "connection_graph.csv"  # Podaj odpowiednią ścieżkę do pliku
    results = analyze_data(file_path)
    print("Wyniki analizy:", results)
