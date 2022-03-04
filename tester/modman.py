import torch
import requests
import json
from os import getpid

debug = False


# Fetch Latest Model Params (StateDict)
def fetch_params(url: str, model_name: str):
    # Send GET request
    r = requests.get(url=url + 'get/' + model_name)

    # Check if Model is Available
    if r.status_code == 404:
        return {}, False

    if r.status_code != 200:
        print("Server Error: Could not fetch Model Parameters.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()

    if debug:
        print("Global Iteration", data['iteration'])
    params = json.loads(data['Params'])
    return params, True


# Fetch Model Lock Status
def model_lock(url: str, model_name: str) -> bool:
    # Send GET request
    r = requests.get(url=url + 'getLockStatus/' + model_name)

    if r.status_code != 200:
        print("Server Error: Could not fetch Lock Status.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()
    return data['LockStatus']


# Send Trained Model Gradients (StateDict)
def send_model_update(url: str, model_name: str, grads: dict):
    body = {'data': {
        'ID': model_name,
        'Gradients': json.dumps(grads)
    }
    }

    # Send POST request
    r = requests.post(url=url + 'updateGradients/' + model_name, json=body)

    if r.status_code != 200:
        print("Server Error: Could not Update Gradients.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()

    return data['reply']


# Send  Model Parameters (StateDict)
def send_model_params(url: str, model_name: str, params: dict, lr: float):
    body = {'data': {
        'ID': model_name,
        'Params': json.dumps(params),
        'LearningRate': str(lr),
    }
    }

    # Send POST request
    r = requests.post(url=url + 'set/', json=body)

    if r.status_code != 200:
        print("Server Error: Could not Update Gradients.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()

    return data['reply']


# Convert State Dict List to Tensor
def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_
