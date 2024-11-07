from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

import yfinance as yf
    
from datetime import datetime,timedelta

from flask_cors import CORS

# Load the trained model and scaler
model = tf.keras.models.load_model('model_2015_2020_Oct_16.keras')

model_EURUSD_CLOSE = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_EURUSD_CLOSE.keras')
model_EURUSD_OPEN = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_EURUSD_OPEN.keras')
model_EURUSD_HIGH = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_EURUSD_HIGH.keras')
model_EURUSD_LOW = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_EURUSD_LOW.keras')

model_JPYUSD_CLOSE = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDJPY_CLOSE.keras')
model_JPYUSD_OPEN = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDJPY_OPEN.keras')
model_JPYUSD_HIGH = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDJPY_HIGH.keras')
model_JPYUSD_LOW = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDJPY_LOW.keras')

model_GBPUSD_CLOSE = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_GBPUSD_CLOSE.keras')
model_GBPUSD_OPEN = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_GBPUSD_OPEN.keras')
model_GBPUSD_HIGH = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_GBPUSD_HIGH.keras')
model_GBPUSD_LOW = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_GBPUSD_LOW.keras')

model_AUDUSD_CLOSE = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_AUDUSD_CLOSE.keras')
model_AUDUSD_OPEN = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_AUDUSD_OPEN.keras')
model_AUDUSD_HIGH = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_AUDUSD_HIGH.keras')
model_AUDUSD_LOW = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_AUDUSD_LOW.keras')

#model_USDCAD_CLOSE = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDCAD_CLOSE.keras')
#model_USDCAD_OPEN = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDCAD_OPEN.keras')
#model_USDCAD_HIGH = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDCAD_HIGH.keras')
#model_USDCAD_LOW = tf.keras.models.load_model('model_2015_2020_200_Epoch_batch_32_seq_60_LSTM_USDCAD_LOW.keras')



model_builder = {
    "EURUSD_CLOSE": model_EURUSD_CLOSE,
    "EURUSD_OPEN": model_EURUSD_OPEN,
    "EURUSD_HIGH": model_EURUSD_HIGH,
    "EURUSD_LOW": model_EURUSD_LOW,
    "JPYUSD_CLOSE": model_JPYUSD_CLOSE,
    "JPYUSD_OPEN": model_JPYUSD_OPEN,
    "JPYUSD_HIGH": model_JPYUSD_HIGH,
    "JPYUSD_LOW": model_JPYUSD_LOW,
    "GBPUSD_CLOSE": model_GBPUSD_CLOSE,
    "GBPUSD_OPEN": model_GBPUSD_OPEN,
    "GBPUSD_HIGH": model_GBPUSD_HIGH,
    "GBPUSD_LOW": model_GBPUSD_LOW,
    "AUDUSD_CLOSE": model_AUDUSD_CLOSE,
    "AUDUSD_OPEN": model_AUDUSD_OPEN,
    "AUDUSD_HIGH": model_AUDUSD_HIGH,
    "AUDUSD_LOW": model_AUDUSD_LOW,
}


#Function to get model from model name
def get_model(model_name):
    print(model_name)
    if model_name in model_builder:
        return model_builder[model_name]
    else:
        return model_builder["EURUSD_CLOSE"]


# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

#Set Sequence Length
sequence_length = 60



#Function to get start date
def get_start_date(date_str):
    date_format = '%Y-%m-%d'
    date_obj = datetime.strptime(date_str, date_format)
    new_date_obj = date_obj - timedelta(days=85)
    new_date_str = new_date_obj.strftime(date_format)
    return new_date_str

#Function to get future dates
def get_future_60_dates(start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    newdates=[]
    cur_date= start_date

    while len(newdates) < 60:
        if cur_date.weekday() < 5:
            newdates.append(cur_date.strftime("%Y-%m-%d"))
        cur_date += timedelta(days=1)
    return newdates


#Test function for percentage change
def calculate_percentage_change(initial_value, final_value):
    try:
        change = final_value - initial_value
        percentage_change = (change / initial_value) * 100
        return percentage_change
    except ZeroDivisionError:
        return "Error."


'''# Function to predict the stock price for a given sequence
def predict_next_stock_price(last_sequence):
    # Normalize the last sequence using the same scaler
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    # Reshape the sequence to match LSTM input shape
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, last_sequence_scaled.shape[0], 1))
    
    # Predict the next stock price
    predicted_price_scaled = model.predict(last_sequence_scaled)
    
    # Inverse transform the prediction to get the actual stock price
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    return predicted_price[0][0]  # Return the predicted value
'''


def predict_future_forex_price_test(last_sequence,days,model_name):
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, last_sequence_scaled.shape[0], 1))
    predictions =[]
    model = get_model(model_name)
    for _ in range(days):
        predicted_price_scaled = model.predict(last_sequence_scaled)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        predictions.append(predicted_price[0][0])

        predicted_price_scaled = np.reshape(predicted_price_scaled, (1, 1, 1))
        last_sequence_scaled = np.append(last_sequence_scaled[:, 1:, :], predicted_price_scaled, axis=1)
    
    return predictions



def predict_next_forex_price_test(data,currency,forex_type):
    currency =currency[:-2]
    data = data[forex_type].values

    new_scaled_data = scaler.transform(data.reshape(-1, 1))
    x_test = []
    y_test = []
    for i in range(sequence_length,len(new_scaled_data)):
        x_test.append(new_scaled_data[i-sequence_length:i,0])
        y_test.append(new_scaled_data[i,0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    new_predictions = 0
    model_name = currency+'_'+forex_type.upper()
    model = get_model(model_name)

    new_predictions = model.predict(x_test)

    '''if currency == 'EURUSD':
        if forex_type == 'Close':
            new_predictions = model_EURUSD_CLOSE.predict(x_test)
        elif forex_type == 'Open':
            new_predictions = model_EURUSD_OPEN.predict(x_test)
        elif forex_type == 'High':
            new_predictions = model_EURUSD_HIGH.predict(x_test)
        else :
            new_predictions = model_EURUSD_LOW.predict(x_test)
    elif currency == 'JPYUSD':
        if forex_type == 'Close':
            new_predictions = model_JPYUSD_CLOSE.predict(x_test)
        elif forex_type == 'Open':
            new_predictions = model_JPYUSD_OPEN.predict(x_test)
        elif forex_type == 'High':
            new_predictions = model_JPYUSD_HIGH.predict(x_test)
        else :
            new_predictions = model_JPYUSD_LOW.predict(x_test)
    elif currency == 'GBPUSD':
        if forex_type == 'Close':
            new_predictions = model_GBPUSD_CLOSE.predict(x_test)
        elif forex_type == 'Open':
            new_predictions = model_GBPUSD_OPEN.predict(x_test)
        elif forex_type == 'High':
            new_predictions = model_GBPUSD_HIGH.predict(x_test)
        else :
            new_predictions = model_GBPUSD_LOW.predict(x_test)
    elif currency == 'AUDUSD':
        if forex_type == 'Close':
            new_predictions = model_AUDUSD_CLOSE.predict(x_test)
        elif forex_type == 'Open':
            new_predictions = model_AUDUSD_OPEN.predict(x_test)
        elif forex_type == 'High':
            new_predictions = model_AUDUSD_HIGH.predict(x_test)
        else :
            new_predictions = model_AUDUSD_LOW.predict(x_test)
    elif currency == 'USDCAD':
        if forex_type == 'Close':
            new_predictions = model_USDCAD_CLOSE.predict(x_test)
        elif forex_type == 'Open':
            new_predictions = model_USDCAD_OPEN.predict(x_test)
        elif forex_type == 'High':
            new_predictions = model_USDCAD_HIGH.predict(x_test)
        else :
            new_predictions = model_USDCAD_LOW.predict(x_test)
    '''
        



    #new_predictions = model_EURUSD_CLOSE.predict(x_test)
    new_predictions = scaler.inverse_transform(new_predictions)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return new_predictions,y_test_actual



def predict_next_forex_price_gain_loss(data,currency):
    currency =currency[:-2]
    data = data['Close'].values

    new_scaled_data = scaler.transform(data.reshape(-1, 1))
    x_test = []
    y_test = []
    for i in range(sequence_length,len(new_scaled_data)):
        x_test.append(new_scaled_data[i-sequence_length:i,0])
        y_test.append(new_scaled_data[i,0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    new_predictions = 0

    if currency == 'EURUSD':
        new_predictions = model_EURUSD_CLOSE.predict(x_test)
    elif currency == 'USDJPY':
        new_predictions = model_JPYUSD_CLOSE.predict(x_test)
    
    #new_predictions = model_EURUSD_CLOSE.predict(x_test)
    new_predictions = scaler.inverse_transform(new_predictions)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return new_predictions,y_test_actual





'''# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    predicted_price =0
    # Get the data from the request
    data = request.json
    print(data['data'])
    
    last_sequence=yf.download(tickers='EURUSD=X',start='2022-05-01',end='2024-10-16',interval ='1d')


    # Extract the last sequence (e.g., last 60 stock prices)
    #last_sequence = np.array(data['last_sequence'])
    
    last_sequence = last_sequence['Close'].values
    last_sequence = last_sequence[-60:]
    # Predict the next stock price
    predicted_price = predict_next_stock_price(last_sequence)

    

    # Return the predicted price as a JSON response
    return jsonify({'predicted_price': str(predicted_price)})
'''

#Route For Future Forecast 
@app.route('/forecastprediction', methods=['POST'])
def testapi():
    data = request.json
    data_currency = data['currency']
    data_forex_type = data['forex_type']
    data_today_date = datetime.today().strftime("%Y-%m-%d")
    data_start_date = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")

    data=yf.download(tickers=data_currency,start=data_start_date,end=data_today_date,interval ='1d')
    first_sequence = data[data_forex_type].values
    first_sequence = first_sequence[:60]
    days = len(data[60:])
    model_name = data_currency[:-2]+'_'+data_forex_type

    #passing data to prediction function
    predicted_price = predict_future_forex_price_test(first_sequence,days,model_name)
    date_label = get_future_60_dates(data_today_date)
    

    # Return the predicted price as a JSON response
    return jsonify({'predicted': str(predicted_price),'date_label':str(date_label)})


#Main Route  
@app.route('/mainprediction', methods=['POST'])
def mainprediction():
  
    data = request.json
    data_currency = data['currency']
    data_start_date = get_start_date(data['start_date'])
    data_end_date = data['end_date']
    data_forex_type = data['forex_type']

    data=yf.download(tickers=data_currency,start=data_start_date,end=data_end_date,interval ='1d')
    #passing data to prediction function
    result = predict_next_forex_price_test(data,data_currency,data_forex_type)

    predicted_result = [item[0] for item in result[0].tolist()]
    actual_result = [item[0] for item in result[1].tolist()]
    date_label = np.array(data.index.strftime('%Y-%m-%d'))
    date_label = date_label[60:].tolist()
  
    # Return the predicted price and Actual price as a JSON response
    return jsonify({'predicted': str(predicted_result),'actual': str(actual_result),'date_label':str(date_label)})


@app.route('/percentagegain',methods=['POST'])
def percentagegain():
    data = request.json
    print(data)
    data_currency = data['currency']
    data_start_date = get_start_date(data['start_date'])
    data_end_date = data['end_date']
    data=yf.download(tickers='EURUSD=X',start=data_start_date,end=data_end_date,interval ='1d')
    result = predict_next_forex_price_gain_loss(data,data_currency)
    predicted_result = [item[0] for item in result[0].tolist()]
    actual_result = [item[0] for item in result[1].tolist()]

    result = calculate_percentage_change(predicted_result[0], actual_result[0])
    final_value=""
    if isinstance(result,str):
        final_value="Error"
    else :
        final_value = "Percentage change: {result:.2f}%"

    return jsonify({'result':final_value})









@app.route('/postcheck', methods=['POST'])
def postcheck():
    req_data = request.get_json()
    return jsonify({'data':req_data})


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)