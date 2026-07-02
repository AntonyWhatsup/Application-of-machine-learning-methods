import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Calculates and prints regression metrics."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Regression Metrics ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

def save_plot(fig, folder, filename):
    """Saves a matplotlib figure to the specific reports directory."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Plot securely saved to {filepath}")