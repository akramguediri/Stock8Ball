import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import matplotlib.pyplot as plt

# 1. Data Collection
def get_stock_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

# 2. Data Preprocessing
def preprocess_data(df, lookback=10):
    df = df[['Close']]  # Select relevant feature(s), consider adding more features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])  # Adjust window size here
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Ensure correct input shape for LSTM
    
    return X, y, scaler

# 3. Model Training
def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Prediction
def predict_future_prices(model, scaler, df, days=5, lookback=10):
    predictions = []
    actual_prices = []
    
    # Save actual prices for the previous `lookback` days
    actual_df = df[['Close']].tail(days + lookback).reset_index(drop=True)
    actual_prices.extend(actual_df.iloc[-lookback:].values.flatten())
    
    last_days = df[-lookback:][['Close']].values
    for _ in range(days):
        last_days_scaled = scaler.transform(last_days)
        X_test = np.array([last_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        predictions.append(predicted_price[0][0])
        
        last_days = np.append(last_days, predicted_price).reshape(-1, 1)
        last_days = last_days[1:]

    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    predictions_df.to_csv('predictions.csv', index=False)
    print(f"Predictions for next {days} days saved to 'predictions.csv'")
    
    return predictions_df

# 5. Plotting Function
def plot_predictions(predictions_df, save_path='predictions_plot.png'):
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], marker='o', linestyle='-', color='r', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved as '{save_path}'")

def collect_data(stock_symbol, start_date, end_date):
    new_data = get_stock_data(stock_symbol, start_date, end_date)
    if os.path.exists('fetched_data.csv'):
        existing_data = pd.read_csv('fetched_data.csv', index_col=0)
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates().reset_index(drop=True)
    else:
        combined_data = new_data
    combined_data.to_csv('fetched_data.csv')
    print("Additional data collected and saved to 'fetched_data.csv'")

def main():
    while True:
        print("1: fetch data\n2: train model\n3: predict for 5 days\n4: collect additional data\n6: plot predicted data\n0: exit")
        choice = input().strip()
        
        if choice == '1':
            stock_symbol = input("Enter the stock symbol: ").strip()
            start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter the end date (YYYY-MM-DD): ").strip()
            df = get_stock_data(stock_symbol, start_date, end_date)
            df.to_csv('fetched_data.csv')
            print("Data fetched and saved to 'fetched_data.csv'")
        elif choice == '2':
            df = pd.read_csv('fetched_data.csv', index_col=0)
            X, y, scaler = preprocess_data(df)
            model = create_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=50, batch_size=32)
            model.save('lstm_model.h5')
            joblib.dump(scaler, 'scaler.gz')
            print("Model trained and saved to 'lstm_model.h5' and scaler saved to 'scaler.gz'")
        elif choice == '3':
            df = pd.read_csv('fetched_data.csv', index_col=0)
            from tensorflow.keras.models import load_model
            model = load_model('lstm_model.h5')
            scaler = joblib.load('scaler.gz')
            predict_future_prices(model, scaler, df)
        elif choice == '4':
            stock_symbol = input("Enter the stock symbol: ").strip()
            start_date = input("Enter the start date for additional data (YYYY-MM-DD): ").strip()
            end_date = input("Enter the end date for additional data (YYYY-MM-DD): ").strip()
            collect_data(stock_symbol, start_date, end_date)
        elif choice == '6':
            predictions_df = pd.read_csv('predictions.csv')
            plot_predictions(predictions_df)
        elif choice == '0':
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 4, 6, or 0.")
ascii_art = """
        ____
    ,dP9CGG88@   ______    _                  __         ____    ______           __   __   
  ,IP  _   Y88 .' ____ \  / |_               [  |  _   .' __ '. |_   _ \         [  | [  |  
 dIi  (_)   G8 | (___ \_|`| |-' .--.   .---.  | | / ]  | (__) |   | |_) |  ,--.   | |  | |  
dCII  (_)   G8  _.____`.  | | / .'`\ \/ /'`\] | '' <   .`____'.   |  __'. `'_\ :  | |  | |  
GCCIi     ,GG8 | \____) | | |,| \__. || \__.  | |`\ \ | (____) | _| |__) |// | |, | |  | |  
GGCCCCCCCGGG88  \______.' \__/ '.__.' '.___.'[__|  \_]`.______.'|_______/ \'-;__/ [___][___] 
GGGGCCCGGGG88888@@@@...                                                                           
Y8GGGGGG8888888@@@@P.....
 Y88888888888@@@@@P......
 `Y8888888@@@@@@@P'......
    `@@@@@@@@@P'.......
        \"\"\"\"........
"""

# Print the ASCII art
print(ascii_art)

if __name__ == "__main__":
    main()
