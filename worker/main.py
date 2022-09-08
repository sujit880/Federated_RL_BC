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
from os import getpid
import os

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
global_reward = [[],[]]
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
        global_reward[0].append(global_score)
        global_reward[1].append(round)
        # honest, malicious_client = vf.verifier(mean_scores=mean_scores)
        honest, malicious_client = vf.verifier_wg(Scores=mean_scores, global_score=global_score)
        total_aggregation_weight = 0.0
        if len(honest)>0:
            print(f'found honest clients......')
            agr_sc = {}
            for client_key in honest:
                aggregation_ratio = 1
                if client_key in clients_verify_stats and client_key in clients_round and client_key in clients_round:
                    res =[x for x, val in enumerate(clients_verify_stats[client_key][1]) if val != 0] #index list of non zero elements
                    aggregation_ratio = (len(clients_round[c_key])/len(res)) #participation ratio
                aggregation_weight = mean_scores[client_key]/(AGGREGATION_SCORE[client_key] +0.0_00_00_00_001) * aggregation_ratio 
                print(f'Calculated aggregation weight: ', aggregation_weight)
                print(f'---->1 : {AGGREGATION_SCORE[c_key]}')
                AGGREGATION_SCORE[client_key] = (aggregation_weight + AGGREGATION_SCORE[client_key])/2.0
                agr_sc[client_key] = aggregation_weight
                
                print(f'---->2 : {AGGREGATION_SCORE[c_key]}')
                params, score = all_params_wscore[client_key]
                all_val_params.append([params,AGGREGATION_SCORE[client_key]])
                print("Valid Params: ", AGGREGATION_SCORE[client_key])
            for c_key in agr_sc:
                aggregation_contribution = agr_sc[c_key]/ np.sum(np.array(list(agr_sc.values())))
                if c_key not in clients_verify_stats: # logging clients reports
                    clients_verify_stats[c_key]=[[aggregation_contribution],[round]]                    
                else:
                    clients_verify_stats[c_key][0].append(aggregation_contribution)
                    clients_verify_stats[c_key][1].append(round)

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
        for c_key in malicious_client:
            if client_key not in clients_verify_stats:# logging clients reports
                clients_verify_stats[c_key]=[[0],[round]]                    
            else:
                clients_verify_stats[c_key][0].append(0)
                clients_verify_stats[c_key][1].append(round)
    sleep(0.2)
print(">>>>>>>>>>>>>>>>>>Training Finished\n Start Logging..............")
stamp = now()
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f'path not exist!!!!!!!!!!!!!')
else: print(f'path exist')
print(f"clients verify stats:", clients_verify_stats)
for c_key in clients_verify_stats:
    print(f'>>>>>>>>>>>>>>>1')
    x = clients_verify_stats[c_key][1]
    y = clients_verify_stats[c_key][0]
    print(f'logging for client {c_key}............')
    plt.plot(x, y)
    plt.ylabel('Aggregation Contribution')
    plt.xlabel('Global rounds')
    plt.grid()
    plt.savefig(f'{log_dir}client_{c_key[4:]}_{stamp.strftime("%d_%m-%H_%M")}.png')
    # plt.title(f' Contribution graph')
    # plt.show()
    plt.close()

    print(f'********************* logging:{c_key} *********************')
    newfilePath = f'{log_dir+"Finished_contri_"+ALIAS}_C_{c_key[-4:]}_{stamp.strftime("%d_%m-%H_%M")}_finished'
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>2')
    with open(f"{newfilePath}.csv", "w") as f:
        print(f'>>>>>>>>>>>>>>>3')
        writer = csv.writer(f)
        writer.writerow(["round","contri"])
        for i in range(len(x)):
            writer.writerow([x[i],y[i]])

global_fig = True
if global_fig:
    print(f'logging for client {c_key}............')
    y,x = global_reward
    plt.plot(x, y)
    plt.ylabel('Mean rewards')
    plt.xlabel('Global rounds')
    plt.savefig(f'{log_dir}global_{stamp.strftime("%d_%m-%H_%M")}.png')
    # plt.title(f' Contribution graph')
    plt.grid()
    plt.show()
    plt.close()
    print(f'********************* logging:global *********************')
    newfilePath = f'{log_dir+"Global"+ALIAS}_{stamp.strftime("%d_%m-%H_%M")}_finished'
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>2')
    with open(f"{newfilePath}.csv", "w") as f:
        print(f'>>>>>>>>>>>>>>>3')
        writer = csv.writer(f)
        writer.writerow(["round","contri"])
        for i in range(len(x)):
            writer.writerow([x[i],y[i]])
