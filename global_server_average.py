# Packages Required
from datetime import datetime

import numpy as np
import tensorflow as tf

import get_local_weight
import upload_to_s3
from logger_config import logger


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


class SimpleMLP:
    @staticmethod
    def build():
        tf.random.set_seed(51)
        np.random.seed(51)
        model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                                                   input_shape=[None]),
                                            tf.keras.layers.SimpleRNN(400, return_sequences=True),
                                            tf.keras.layers.SimpleRNN(400), tf.keras.layers.Dense(1), ])
        return model


def run_global_server():
    # initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build()
    # commence global training loop
    comms_round = 100
    FRA_URL = "http://10.192.20.110:8000/local_training"
    PARIS_URL = "http://192.168.11.4:8010/local_training"
    STHLM_URL = "http://10.0.10.75:8020/local_training"
    model_upload_bucket_name = 'fra-5g-nw-global'

    # Get today's date in the format YYYY-MM-DD
    current_datetime = datetime.now()
    # Format the date and time as a string (YYYY-MM-DD_HH-MM-SS)
    formatted_datetime = current_datetime.strftime('%d-%m-%Y_%H:%M:%S')
    directory_name = f'{formatted_datetime}'
    model_upload_bucket_name = 'fra-5g-nw-global'

    for comm_round in range(comms_round):
        # get global weights
        global_weights = global_model.get_weights()
        Local_FRA = get_local_weight.get_weights(FRA_URL, global_weights)
        logger.info(f"Loaded response from Client - {Local_FRA[0]}")
        Local_PARIS = get_local_weight.get_weights(PARIS_URL, global_weights)
        logger.info(f"Loaded response from Client - {Local_PARIS[0]}")
        LOCAL_STHLM = get_local_weight.get_weights(STHLM_URL, global_weights)
        logger.info(f"Loaded response from Client - {LOCAL_STHLM[0]}")
        Clients = [Local_FRA]
        Client_name = [i[0] for i in Clients]
        Client_length = [i[1] for i in Clients]
        Client_weights = [i[2] for i in Clients]

        global_length = sum(Client_length)

        scaling_factor = [x / global_length for x in Client_length]

        scaled_local_weights_list = []
        for i in range(0, len(Client_weights)):
            scaled_weights = scale_model_weights(Client_weights[i], scaling_factor[i])
            scaled_local_weights_list.append(scaled_weights)

        average_weights = sum_scaled_weights(scaled_local_weights_list)

        global_model.set_weights(average_weights)
        name = "avg_model" + str(comm_round) + ".h5"
        logger.info(f"Round number {comm_round} completed!")
        logger.info(f"Saving file - {name}")
        global_model.save(name)
        upload_to_s3.upload_file_to_s3(model_upload_bucket_name, name, directory_name)
    logger.info(f"Completed all {comms_round} of training")
