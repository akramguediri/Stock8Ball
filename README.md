```plaintext
        ____
    ,dP9CGG88@   ______    _                  __         ____    ______           __   __   
  ,IP  _   Y88 .' ____ \  / |_               [  |  _   .' __ '. |_   _ \         [  | [  |  
 dIi  (_)   G8 | (___ \_|`| |-' .--.   .---.  | | / ]  | (__) |   | |_) |  ,--.   | |  | |  
dCII  (_)   G8  _.____`.  | | / .'`\ \/ /'`\] | '' <   .`____'.   |  __'. `'_\ :  | |  | |  
GCCIi     ,GG8 | \____) | | |,| \__. || \__.  | |`\ \ | (____) | _| |__) |// | |, | |  | |  
GGCCCCCCCGGG88  \______.' \__/ '.__.' '.___.'[__|  \_]`.______.'|_______/ \'-;__/[___][___] 
GGGGCCCGGGG88888@@@@...                                                                           
Y8GGGGGG8888888@@@@P.....
 Y88888888888@@@@@P......
 `Y8888888@@@@@@@P'......
    `@@@@@@@@@P'.......
        \"\"\"\"........
```
# Stock Price Prediction with LSTM

This project utilizes LSTM (Long Short-Term Memory) neural networks to predict future stock prices based on historical data.

## Overview

The project is structured into several main components:

1. **Data Collection:** Fetch historical stock price data using Yahoo Finance API (`yfinance`).
2. **Data Preprocessing:** Prepare the data by scaling and structuring it for LSTM model input.
3. **Model Training:** Train an LSTM model to learn from historical data.
4. **Prediction:** Use the trained model to predict future stock prices.
5. **Additional Data Collection:** Optionally collect more data to update the model.
6. **Plotting:** Visualize the predicted prices using matplotlib.

## Components

- **`get_stock_data(stock_symbol, start_date, end_date)`**: Function to fetch historical stock data.
- **`preprocess_data(df, lookback=10)`**: Function to preprocess data for LSTM model.
- **`create_lstm_model(input_shape, units=50, dropout_rate=0.2)`**: Function to define the LSTM model architecture.
- **`predict_future_prices(model, scaler, df, days=5, lookback=10)`**: Function to predict future stock prices.
- **`collect_data(stock_symbol, start_date, end_date)`**: Function to collect additional historical data.
- **`plot_predictions(predictions_df, save_path='predictions_plot.png')`**: Function to plot predicted prices.

## Usage

1. **Setup:**
   - Ensure Python 3.x is installed.
   - Install required packages using `pip install -r requirements.txt`.

2. **Data Collection:**
   - Run the script and choose option 1 to fetch historical data for a stock symbol within a specified date range.

3. **Model Training:**
   - After collecting data, choose option 2 to preprocess the data, train the LSTM model, and save it.

4. **Prediction:**
   - Choose option 3 to load the trained model and scaler, then predict future stock prices for the next 5 days.

5. **Additional Data Collection:**
   - Choose option 4 to collect more historical data for updating the model.

6. **Plotting:**
   - Choose option 6 to plot the predicted stock prices saved in 'predictions.csv'.

## Files

- **`main.py`**: Main script to run the entire project.
- **`lstm_model.h5`**: Saved LSTM model file.
- **`scaler.gz`**: Saved MinMaxScaler object for data scaling.
- **`fetched_data.csv`**: CSV file containing fetched historical stock data.
- **`predictions.csv`**: CSV file containing predicted stock prices for future dates.
- **`predictions_plot.png`**: Plot of predicted stock prices.

## Dependencies

- `yfinance` for fetching stock data.
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `scikit-learn` for data preprocessing.
- `tensorflow` for building and training the LSTM model.
- `matplotlib` for plotting predicted prices.

## Notes

- Ensure an internet connection to fetch data from Yahoo Finance.
- The model's accuracy may vary based on data quality and chosen parameters.
