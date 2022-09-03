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

ALL_CLIENTS = {}
AGGREGATION_SCORE = {}
numberof_appearance = {}



while True:
    if modman.get_update_lock(URL, ALIAS):
        all_params_wscore, global_params = modman.get_client_params(URL, ALIAS)
        all_val_params=[]
        mean_scores={}
        keys = all_params_wscore.keys()
        for c_key in keys:
            params, score = all_params_wscore[c_key]
            key, score = ts.Test_Params(params=params, client_key=c_key)
            # all_params.append([json.loads(params),score])
            mean_scores[key] = ts.Test_Params(params=params, client_key=c_key)
        global_score = ts.Test_Params(params=global_params, client_key="global")
        # honest, malicious_client = vf.verifier(mean_scores=mean_scores)
        honest, malicious_client = vf.verifier_wg(mean_scores=mean_scores, global_score=global_score)
        total_aggregation_weight = 0.0
        if len(honest)>0:
            for client_key in honest:
                total_aggregation_weight += AGGREGATION_SCORE[client_key]/len(honest)
            for client_key in honest:
                aggregation_weight = mean_scores[client_key]/(total_aggregation_weight+0.0_00_00_00_001)
                AGGREGATION_SCORE[client_key] += aggregation_weight 
                params, score = all_params_wscore[client_key]
                all_val_params.append([params,mean_scores[client_key]])

            ##############################
            # Aggregate all valid params  
            #############################
            print("Number of honest client", len(honest))  
            average_params=works.Federated_average(all_val_params)
            # print("\n params type: ", type(average_params))
            # print("Aggregated Params:->\n", average_params)
            modman.send_global_model_update(URL,ALIAS, modman.convert_tensor_to_list(average_params))
            print("Update Complete . . . .")
        else:
            modman.send_global_model_update(URL,ALIAS, modman.convert_tensor_to_list(global_params))

    sleep(0.2)
