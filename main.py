import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def preprocess_data(raw_data):
    # Drop unnecessary columns
    processed_data = raw_data.drop(columns=['device_id', 'latitude', 'longitude'])

    # Drop rows with invalid data
    processed_data = processed_data.dropna()
    processed_data = processed_data.replace(0, pd.NA)
    processed_data = processed_data.mask(processed_data.eq(processed_data.iloc[0]).all(axis=1)).dropna()

    # Convert 'timestamp' to datetime
    processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])

    # Drop rows with constant values for more than 5 minutes
    timestamp_diff = processed_data['timestamp'].diff()
    invalid_rows = timestamp_diff.dt.total_seconds() < 300  # 300 seconds = 5 minutes
    processed_data = processed_data[~invalid_rows]

    processed_data.to_csv("imputed_dataset.csv", index=False)

def impute_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    parameters_to_impute = ['temperature', 'humidity', 'pm2_5', 'pm10']
    data[parameters_to_impute] = imputer.fit_transform(data[parameters_to_impute])

    return data

def create_sequences(data, sequence_length=10):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data.iloc[i:i+sequence_length]
        sequences.append(sequence.values)
    return np.array(sequences)

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, rmse, mae, model

def main():
    # Read the original dataset
    raw_data = pd.read_csv("air_quality_1.csv")  # Replace with your actual file path

    # Step 1: Preprocess the data
    preprocess_data(raw_data)

    processed_data = pd.read_csv("imputed_dataset.csv")

    # Step 2: Create sequences
    sequence_length = 10
    sequences = create_sequences(processed_data, sequence_length)

    # Step 3: Impute missing values in sequences
    imputer = SimpleImputer(strategy='mean')
    parameters_to_impute = ['temperature', 'humidity', 'pm2_5', 'pm10']
    sequences[:, :, 2:6] = imputer.fit_transform(sequences[:, :, 2:6].reshape(-1, 4)).reshape(-1, sequence_length, 4)

    # Step 4: Train and evaluate the model
    X = sequences[:, :, 2:6]
    y_pm2_5 = sequences[:, -1, 2]
    y_pm10 = sequences[:, -1, 3]
    y_humidity = sequences[:, -1, 1]
    y_temperature = sequences[:, -1, 0]

    r2_pm2_5, rmse_pm2_5, mae_pm2_5, model_pm2_5 = train_and_evaluate_model(X, y_pm2_5)
    r2_pm10, rmse_pm10, mae_pm10, model_pm10 = train_and_evaluate_model(X, y_pm10)
    r2_humidity, rmse_humidity, mae_humidity, model_humidity = train_and_evaluate_model(X, y_humidity)
    r2_temperature, rmse_temperature, mae_temperature, model_temperature = train_and_evaluate_model(X, y_temperature)

    # Print or save the model if needed
    # model_pm2_5.save('lstm_model_pm2_5.h5')
    # (Repeat for other models)

    # Output test statistics
    with open("test_statistics_lstm.txt", "w") as f:
        f.write("pm2_5 Statistics:\n")
        f.write(f"R2 Score: {r2_pm2_5}\n")
        f.write(f"RMSE: {rmse_pm2_5}\n")
        f.write(f"MAE: {mae_pm2_5}\n")

        f.write("pm10 Statistics:\n")
        f.write(f"R2 Score: {r2_pm10}\n")
        f.write(f"RMSE: {rmse_pm10}\n")
        f.write(f"MAE: {mae_pm10}\n")

        f.write("humidity Statistics:\n")
        f.write(f"R2 Score: {r2_humidity}\n")
        f.write(f"RMSE: {rmse_humidity}\n")
        f.write(f"MAE: {mae_humidity}\n")

        f.write("temperature Statistics:\n")
        f.write(f"R2 Score: {r2_temperature}\n")
        f.write(f"RMSE: {rmse_temperature}\n")
        f.write(f"MAE: {mae_temperature}\n")

if __name__ == "__main__":
    main()