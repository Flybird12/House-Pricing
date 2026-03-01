import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("train.csv")

# Select strong numerical features
features = [
    "GrLivArea",      # Living Area
    "BedroomAbvGr",   # Bedrooms
    "FullBath",       # Bathrooms
    "OverallQual",    # Overall Quality
    "GarageCars",     # Garage Capacity
    "TotalBsmtSF"     # Basement Area
]

target = "SalePrice"

data = df[features + [target]].dropna()

X = data[features]
y = np.log(data[target])   # Log transform improves performance

# -------------------------
# 2. Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Build Pipeline (Scaling + Model)
# -------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001)
}

results = {}

for name, model in models.items():
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    # Cross Validation
    cv_score = cross_val_score(
        pipeline, X_train, y_train,
        scoring="r2", cv=5
    ).mean()
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = (cv_score, r2, mse)

# -------------------------
# 4. Print Results
# -------------------------
for model_name, (cv, r2, mse) in results.items():
    print(f"\n{model_name}")
    print("Cross-Validation R2:", round(cv, 4))
    print("Test R2:", round(r2, 4))
    print("Test MSE:", round(mse, 4))

# -------------------------
# 5. Visualizations for Linear Regression
# -------------------------
# Use Linear Regression for detailed plots
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# Convert log predictions back to original scale for plotting
y_test_actual = np.exp(y_test)
y_pred_actual = np.exp(y_pred_lr)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted (Scatter)
ax1 = axes[0, 0]
ax1.scatter(y_test_actual, y_pred_actual, alpha=0.5, edgecolors='k', linewidths=0.5)
ax1.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Sale Price ($)', fontsize=12)
ax1.set_ylabel('Predicted Sale Price ($)', fontsize=12)
ax1.set_title('Linear Regression: Actual vs Predicted', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test_actual - y_pred_actual
ax2 = axes[0, 1]
ax2.scatter(y_pred_actual, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Sale Price ($)', fontsize=12)
ax2.set_ylabel('Residuals ($)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison (R2 Scores)
ax3 = axes[1, 0]
model_names = list(results.keys())
cv_scores = [results[name][0] for name in model_names]
test_r2_scores = [results[name][1] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax3.bar(x - width/2, cv_scores, width, label='Cross-Val R2', color='steelblue')
bars2 = ax3.bar(x + width/2, test_r2_scores, width, label='Test R2', color='coral')

ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('R2 Score', fontsize=12)
ax3.set_title('Model Comparison: R2 Scores', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

# Plot 4: Price Distribution (Actual vs Predicted)
ax4 = axes[1, 1]
ax4.hist(y_test_actual, bins=30, alpha=0.5, label='Actual Prices', color='blue', edgecolor='black')
ax4.hist(y_pred_actual, bins=30, alpha=0.5, label='Predicted Prices', color='orange', edgecolor='black')
ax4.set_xlabel('Sale Price ($)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Price Distribution: Actual vs Predicted', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved to 'house_price_analysis.png'")

