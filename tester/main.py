# Import Libraries
import modman
import test_score as ts
from time import sleep
import json

ALIAS = 'experiment_01'
ENV_NAME = 'CartPole-v0'

# API endpoint
URL = "http://localhost:3000/api/model/"


# Testing All Clients Parameters
while True:
    if modman.get_update_lock(URL, ALIAS):
        all_params_wscore = modman.get_client_params(URL, ALIAS)
        Scores={}
        keys = all_params_wscore.keys()
        for c_key in keys:
            params, old_score = all_params_wscore[c_key]
            key, score = ts.Test_Params(params=json.loads(params), client_key=c_key) 
            Scores[key] = score
        print("...all scores...\n\n", Scores, "\n\n...****.....\n" )   
        modman.send_score(URL,ALIAS,Scores)
        print("Update Scores Complete . . . .")

    sleep(0.2)

