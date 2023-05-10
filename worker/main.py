from builtins import type
import modman
import works
import json
import numpy as np
import torch
from time import sleep

# Model Name / ALIAS (in Client)
ALIAS = 'sbac1_3'

# API endpoint
URL = "http://localhost:3000/api/model/"

# URL = "http://localhost:5500/api/model/"

while True:
    if modman.get_update_lock(URL, ALIAS):
        all_params_wscore = modman.get_client_params(URL, ALIAS)
        all_params=[]
        keys = all_params_wscore.keys()
        for key in keys:
            params, score = all_params_wscore[key]
            all_params.append([json.loads(params),score])
        average_params=works.Federated_average(all_params)
        print("\n params type: ", type(average_params))
        print("Aggregated Params:->\n", average_params)
        modman.send_global_model_update(URL,ALIAS, modman.convert_tensor_to_list(average_params))
        print("Update Complete . . . .")

    sleep(0.2)
