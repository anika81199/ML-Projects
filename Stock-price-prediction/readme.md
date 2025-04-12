# Stock Price Prediction using LSTM

This project demonstrates how to build and train a Long Short-Term Memory (LSTM) neural network model to predict stock prices. The example uses Apple Inc. (AAPL) stock data.

## Goal

The primary goal is to predict the future closing price of a stock based on its historical price data using an LSTM model, which is well-suited for sequence prediction tasks.

## Data Source

*   Historical stock price data is downloaded directly from Yahoo Finance using the `yfinance` library.
*   The example uses AAPL stock data from January 1, 2010, to January 12, 2024.

## Libraries Used

*   **Data Handling:** `numpy`, `pandas`
*   **Data Acquisition:** `yfinance`
*   **Visualization:** `matplotlib`
*   **Machine Learning:**
    *   `scikit-learn`: `MinMaxScaler` for data normalization.
    *   `tensorflow.keras`: `Sequential` for model building, `LSTM`, `Dense`, `Dropout` for layers.

## Methodology

1.  **Data Acquisition:** Download historical stock data for the specified ticker symbol (e.g., AAPL) and date range.
2.  **Data Preparation:**
    *   Select the 'Close' price for prediction.
    *   Normalize the closing prices using `MinMaxScaler` to scale values between 0 and 1.
    *   Create sequences of historical data (e.g., using the past 60 days' prices to predict the next day's price).
    *   Split the data into training (80%) and testing (20%) sets.
3.  **Model Building:**
    *   Construct a Sequential Keras model.
    *   Add LSTM layers with Dropout for regularization.
    *   Add Dense layers for processing.
    *   Include a final Dense layer with one unit for the predicted price output.
4.  **Model Training:**
    *   Compile the model using the 'adam' optimizer and 'mean_squared_error' as the loss function.
    *   Train the model on the training data for a specified number of epochs (e.g., 50).
    *   Monitor training and validation loss.
5.  **Evaluation & Prediction:**
    *   Use the trained model to predict prices on the test set.
    *   Inverse transform the predicted (scaled) prices back to their original scale.
    *   Visualize the actual vs. predicted prices to evaluate model performance.
    *   Visualize the trend comparison (increase/decrease) between actual and predicted prices.
6.  **Model Saving:**
    *   Save the trained model to a file (`lstm_stock_predictor.h5`) for later use.
    *   Demonstrate how to load the saved model and make predictions on new data.

## Results

The notebook includes visualizations comparing the actual stock prices against the prices predicted by the LSTM model on the test data. It also shows a comparison of the predicted price movement trends. The trained model is saved as `lstm_stock_predictor.h5`.

## How to Use

1.  Ensure all required libraries are installed (`pip install yfinance numpy pandas matplotlib scikit-learn tensorflow`).
2.  Run the Jupyter Notebook (`stock_price_prediction_model.ipynb`).
3.  Modify the `stock_symbol`, `start`, and `end` dates as needed.
4.  The trained model will be saved as `lstm_stock_predictor.h5`. You can load this model later for predictions without retraining.
