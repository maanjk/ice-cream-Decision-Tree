import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("Ice Cream.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df[["Temperature"]]
    y = df["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2, X_train, y_train

df = load_data()
model, rmse, r2, X_train, y_train = train_model(df)

st.title("Ice Cream Revenue Prediction (Decision Tree Regression)")
st.write("Dataset: Ice Cream Sales (Temperature → Revenue)")

st.subheader("Model performance on test set")
st.write(f"RMSE: **{rmse:.2f}**")
st.write(f"R²: **{r2:.3f}**")

st.subheader("Predict revenue for a given temperature")

temp_min = float(df["Temperature"].min())
temp_max = float(df["Temperature"].max())
temp_default = float(df["Temperature"].mean())

temperature = st.slider(
    "Temperature (°C)", temp_min, temp_max, temp_default, 0.5
)

if st.button("Predict revenue"):
    x = np.array([[temperature]])
    revenue_pred = model.predict(x)[0]
    st.success(f"Predicted revenue: **{revenue_pred:.2f}**")

st.subheader("Training data and Decision Tree fit")

show_plot = st.checkbox("Show scatter + tree prediction curve", value=True)
if show_plot:
    x_grid = np.linspace(temp_min, temp_max, 300).reshape(-1, 1)
    y_grid = model.predict(x_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Temperature"], df["Revenue"], color="red", label="Data")
    ax.plot(x_grid, y_grid, color="blue", label="Tree prediction")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Revenue")
    ax.legend()
    st.pyplot(fig)