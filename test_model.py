
import tensorflow as tf
import numpy as np
import csv
import download_from_s3
from logger_config import logger
import pandas as pd

def windowed_dataset(client_data, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(client_data)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def ts_mlflow_test(data,model, in_look_back, in_batch_size, df, comm_round):
    tf.random.set_seed(7)
    dataset = data
    #dataset = windowed_dataset(data, 20, 16, len(data))
    window_size = 20
    series_trans = dataset
    forecast = []
    for time in range(len(series_trans)-window_size):
        forecast.append(model.predict(series_trans[time:time + window_size][np.newaxis]))
    forecast = forecast[:]
    results = np.array(forecast)[:, 0, 0]
    MSE = tf.keras.metrics.mean_squared_error(series_trans[window_size:len(series_trans)], results).numpy()
    MAPE = tf.keras.metrics.mean_absolute_percentage_error(series_trans[window_size:len(series_trans)], results).numpy()
    T=[]
    a = 200
    for i in range(a):
        T.append(forecast[i][0][0])
    logger.info('Train Score: %.2f RMSE' % (MSE))
    logger.info('Train Score: %.2f MAPE' % (100 - MAPE))
    df.loc[len(df)] = {'comm_rounds': comm_round, 'RMSE': MSE, 'MAPE': (100 - MAPE)}
    logger.info(f"Added to dataframe - comm_rounds: {comm_round}, RMSE: {MSE}, MAPE: {(100 - MAPE)}")
    return df

def test_model(type):
    Mbits_transmitted_test = []
    comm_rounds=100

    # Define an empty dictionary with column names
    columns = {
        'comm_rounds': [],
        'RMSE': [],
        'MAPE': []
    }

    region="fra"
    # Create the empty DataFrame
    df = pd.DataFrame(columns)
    s3_object_key = f'test_model/{region}/Traffic_Test_Data.csv'
    download_from_s3.download_file_from_s3("fra-5g-nw-global",s3_object_key,"Traffic_Test_Data.csv")

    # For testing the performance of the model.
    with open('./Traffic_Test_Data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            Mbits_transmitted_test.append(float(row[1]))
    # converting the lists into arrays
    data = np.array(Mbits_transmitted_test)
    from keras.models import load_model
    for comm_round in range(0, comm_rounds):
        name = str(type) + "_model" + str(comm_round) + ".h5"
        logger.info(f"Reading file - {name}")
        model = load_model(name)
        df = ts_mlflow_test(data, model, 20, 16, df, comm_round)
    file_path = f'test_score_{type}.csv'
    logger.info(f"Adding test result to file - {file_path}")
    # Write the DataFrame to the CSV file
    df.to_csv(file_path, index=False)