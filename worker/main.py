from builtins import type
import modman
import works
import json
import numpy as np
import torch
import test_score as ts
import verifyer as vf
from time import sleep
import matplotlib.pyplot as plt
import csv
import datetime

now = datetime.datetime.now

# Model Name / ALIAS (in Client)
ALIAS = 'experiment_01'

# API endpoint
URL = "http://localhost:3000/api/model/"

# URL = "http://localhost:5500/api/model/"

ALL_CLIENTS = {}
AGGREGATION_SCORE = {}
numberof_appearance = {}

round = 0
clients_verify_stats = {}
clients_round = {}

COMPLETE = False

while True:
    if COMPLETE: break  ##Stop training if model converged or reached maximum epochs in any clients.
    lock_status = modman.get_update_lock(URL, ALIAS)
    if lock_status[1]: COMPLETE = True
    if lock_status[0]:
        all_params_wscore, global_params = modman.get_client_params(URL, ALIAS)
        round +=1
        all_val_params=[]
        mean_scores={}
        keys = all_params_wscore.keys()
        for c_key in keys:
            if c_key not in clients_round: #adding how many round one client participated
                clients_round[c_key] = [round]
            else:
                clients_round[c_key].append(round)
            params, score = all_params_wscore[c_key]
            key, score = ts.Test_Params(params=params, client_key=c_key)
            if c_key not in AGGREGATION_SCORE:
                print("\nGot update from new clients.............\n")
                AGGREGATION_SCORE[c_key] = score
            # all_params.append([json.loads(params),score])
            mean_scores[key] = score
        _, global_score = ts.Test_Params(params=global_params, client_key="global")
        # honest, malicious_client = vf.verifier(mean_scores=mean_scores)
        honest, malicious_client = vf.verifier_wg(Scores=mean_scores, global_score=global_score)
        total_aggregation_weight = 0.0
        if len(honest)>0:
            print(f'found honest clients......')
            for client_key in honest:
                aggregation_ratio = 1
                if client_key in clients_verify_stats:
                    aggregation_ratio = (len(clients_round[c_key])/len(clients_verify_stats[client_key][1]))
                aggregation_weight = mean_scores[client_key]/(AGGREGATION_SCORE[client_key] +0.0_00_00_00_001) * aggregation_ratio 
                print(f'Calculated aggregation weight: ', aggregation_weight)
                print(f'---->1 : {AGGREGATION_SCORE[c_key]}')
                AGGREGATION_SCORE[client_key] = (aggregation_weight + AGGREGATION_SCORE[client_key])/2.0
                if client_key not in clients_verify_stats: # logging clients reports
                    clients_verify_stats[client_key]=[[aggregation_weight],[round]]                    
                else:
                    clients_verify_stats[client_key].append([[aggregation_weight],[round]])
                print(f'---->2 : {AGGREGATION_SCORE[c_key]}')
                params, score = all_params_wscore[client_key]
                all_val_params.append([params,AGGREGATION_SCORE[client_key]])
                print("Valid Params: ", AGGREGATION_SCORE[client_key])

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
            print(f'no honest clients detected........')
            modman.send_global_model_update(URL,ALIAS, global_params)

    sleep(0.2)
stamp = now()
newfilePath = f'{log_dir+str(getpid())+":Finished"+ALIAS}_{stamp.strftime("%d_%m_%Y-%H_%M_%S")}_finished'
rows = zip(clients_verify_stats.values())
with open(newfilePath, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
for c_key in clients_verify_stats:
    y,x = clients_verify_stats[client_key]
    plt.plot(x, y)
    plt.ylabel('Aggregation Contribution')
    plt.xlabel('Global rounds')
    plt.savefig(f'./logs/{client_key}_{ALIAS}.png')
    plt.title(f'{client_key}'s Contribution graph !')
    plt.show()
    plt.close()
