from builtins import type
import modman
import works
import json
import numpy as np
import torch
import test_score as ts
import verifyer as vf
from time import sleep

# Model Name / ALIAS (in Client)
ALIAS = 'experiment_01'

# API endpoint
URL = "http://localhost:3000/api/model/"

# URL = "http://localhost:5500/api/model/"

while True:
    if modman.get_update_lock(URL, ALIAS):
        all_params_wscore = modman.get_client_params(URL, ALIAS)
        all_val_params=[]
        Scores={}
        keys = all_params_wscore.keys()
        for c_key in keys:
            params, score = all_params_wscore[c_key]
            key, score = ts.Test_Params(params=json.loads(params), client_key=c_key)
            # all_params.append([json.loads(params),score])
            Scores[key] = score
        honest, malicious_client = vf.verifier(Scores=Scores)
        for client_key in honest:
            params, score = all_params_wscore[client_key]
            all_val_params.append([json.loads(params),Scores[client_key]])

        ##############################
        # Aggregate all valid params  
        #############################
        print("Number of honest client", len(honest))  
        average_params=works.Federated_average(all_val_params)
        # print("\n params type: ", type(average_params))
        # print("Aggregated Params:->\n", average_params)
        modman.send_global_model_update(URL,ALIAS, modman.convert_tensor_to_list(average_params))
        print("Update Complete . . . .")

    sleep(0.2)
