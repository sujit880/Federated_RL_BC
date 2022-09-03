import requests
import json

from typing import List
import torch
import ipfshttpclient as ipfs

## Create IPFS Client
client = ipfs.connect("/ip4/172.16.26.15/tcp/5001/http")

try_cnt = 0

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


#################################################################
# Network Handlers
#################################################################

# Collect Client Params

def get_client_params(url: str, model_name: str):

    r = requests.get(url=url+'getclientP/'+model_name)
    if r.status_code != 200:
        print("Server Error: Could not fetch all params.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()

    print('data reply', data)

    clients = data["clients"]
    clients_hash = clients.keys()
    clients_params ={}
    for x in clients_hash:
        value1 = client.get_json(x)
        client_params =json.loads(value1["params"])
        # print("Params:", value)
        clients_params[clients[x][1]]=[client_params,clients[x][0]] #clients[x][1] -> clients_key
    # global_params = client.cat(data["global_params"]) 
    value1 = client.get_json(data["global_params"])
    global_params =json.loads(value1["params"])
    return clients_params, global_params;



# Send Trained Model Gradients (StateDict)
def send_global_model_update(url: str, model_name: str, params: dict):

    #Upload parameters to IPFS
    parameters = {"params": json.dumps(params)}
    params_hash = client.add_json(parameters)
    
    # body = {'data': {
    #     'Params': params_hash,
    #     'report': report
    # }
    # }

    body = {'data': {
        'Params': params_hash,
    }
    }

    # Send POST request
    r = requests.post(url=url + 'applyUpdate/' + model_name, json=body)

    if r.status_code != 200:
        print("Server Error: Could not Apply Model Updates.\nQuitting...")
        quit()

    # Extract data in json format
    data = r.json()
    print( data['reply'])

    return data['reply']

# Fetch Model Lock Status  
def get_update_lock(url: str, model_name: str):
    global try_cnt
    try:
        r = requests.get(url=url + 'getUlock/'+model_name)
        if r.status_code != 200:
            print("Server Error: Could not fetch Lock Status.\nQuitting...")
            # quit()

        # Extract data in json format
        data = r.json()
        print("data->", data)
        print("Lock data:->", data['LockStatus'])
        try_cnt = 0
        return data['LockStatus'] 
    except:
        try_cnt += 1
        print("Server error try again after 2 sec.")
        if try_cnt >20:
            print("Couldn't connect to server quiting")
            quit()
        sleep(2)



