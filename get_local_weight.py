import requests
import pickle
import base64


def get_weights(url, global_weights):
    json_response = []
    try:
        serialized_weights = pickle.dumps(global_weights)
        base64_encoded_weights = base64.b64encode(serialized_weights).decode('utf-8')
        headers = {'Content-Type': 'application/json'}
        data = {'weights': base64_encoded_weights}
        print(f"Fetching Response from {url}")
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # Parse the JSON response
            json_response = response.json()
        else:
            print("Response code is not 200")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    serialized_weights = base64.b64decode(json_response[2])
    print(f"Got response from {json_response[0]}")
    weight = pickle.loads(serialized_weights)
    local_result = json_response[0], json_response[1], weight
    return local_result
