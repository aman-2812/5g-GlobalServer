# Packages Required
import tensorflow as tf
import numpy as np
import get_local_weight
import upload_to_s3


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def median_weights(weight_list):
    '''Return the median of weights. '''

    avg_grad = list()

    for grad_list_tuple in zip(*weight_list):
        layer_mean = np.median(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

class SimpleMLP:
    @staticmethod
    def build():
        tf.random.set_seed(51)
        np.random.seed(51)
        model = tf.keras.models.Sequential([ tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                           input_shape=[None]), tf.keras.layers.SimpleRNN(400, return_sequences=True),
                                           tf.keras.layers.SimpleRNN(400), tf.keras.layers.Dense(1), ])
        return model


def run_global_server():
    # initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build()
    # commence global training loop
    comms_round = 4
    FRA_URL = "http://52.59.221.4:8000/local_training"
    PARIS_URL = "http://15.237.196.132:8010/local_training"
    STHLM_URL = "http://16.170.148.134:8020/local_training"
    for comm_round in range(comms_round):
        # get global weights
        global_weights = global_model.get_weights()
        Local_FRA = get_local_weight.get_weights(FRA_URL, global_weights)
        print(f"Loaded response from Client - {Local_FRA[0]}")
        Local_PARIS = get_local_weight.get_weights(PARIS_URL, global_weights)
        print(f"Loaded response from Client - {Local_PARIS[0]}")
        LOCAL_STHLM = get_local_weight.get_weights(STHLM_URL, global_weights)
        print(f"Loaded response from Client - {LOCAL_STHLM[0]}")
        Clients = [Local_FRA, Local_PARIS, LOCAL_STHLM]
        Client_name = [i[0] for i in Clients]
        Client_length = [i[1] for i in Clients]
        Client_weights = [i[2] for i in Clients]

        global_length = sum(Client_length)

        scaling_factor = [1, 1, 1]

        scaled_local_weights_list = []
        for i in range(0, len(Client_weights)):
            scaled_weights = scale_model_weights(Client_weights[i], scaling_factor[i])
            scaled_local_weights_list.append(scaled_weights)

        average_weights = median_weights(scaled_local_weights_list)

        global_model.set_weights(average_weights)
        name = "median_model" + str(comm_round) + ".h5"
        print(f"Round number {comm_round}")
        global_model.save(name)
        upload_to_s3.upload_file_to_s3(name)
    print(f"Completed all {comms_round} rounds of training")