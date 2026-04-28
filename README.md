# Simple ML Tasks – Repository

This repository contains **three Jupyter notebooks** that demonstrate basic concepts of **data analysis, visualization, and regression models in Python**.

## Files Overview

### 1. `task1_iris_visualization.ipynb`

This notebook explores the **Iris dataset**.
It loads the dataset using pandas and performs basic inspection using functions like `head()`, `info()`, and `describe()`.
It also visualizes the data using:

* Scatter plots to show relationships between features
* Histograms to show value distributions
* Box plots to detect possible outliers

---

### 2. `task2_stock_prediction.ipynb`

This notebook demonstrates a **simple stock price prediction task**.
It downloads historical stock data using the `yfinance` library and uses features like **Open, High, Low, and Volume** to predict the **Close price**.

A **Linear Regression model** is trained and the results are visualized by plotting **actual vs predicted stock prices**.

---

### 3. `task6_house_price_prediction.ipynb`

This notebook builds a **basic house price prediction model**.
A small dataset with features such as **house size, number of bedrooms, and location score** is used.

A **Linear Regression model** is trained and evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

The notebook also plots **actual vs predicted house prices**.

---

These notebooks are meant to demonstrate **simple machine learning workflows**, including:

* Loading and inspecting datasets
* Data visualization
* Training regression models
* Evaluating predictions
