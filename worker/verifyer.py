import numpy as np
import random
sigma=0.85 #Hyperparameter to verify score
# Scores = {}
# Scores['key1']= 125
# Scores['key2']= 170
# Scores['key3']= 180
# Scores['key4']= 190
# Scores['key5']= 125
# Scores['key6']= 125
# Scores['key7']= 125


#### Compare global round versus aggregation weight or contribution.


## Verify If Client Is malicious

def verifier(Scores):
    rand = random.random()
    if rand<0.3: 
        sigma = 0.3
    elif rand>0.89:
        sigma = 0.9
    else:
        sigma = 0.1 + rand
    keys = list(Scores.keys())
    values = list(Scores.values())
    print("Keys: ", keys, "Values: ", values)
    score_mean = np.mean(np.array(values))
    print("Mean: ",score_mean)
    malicious = []
    honest=[]
    for index in range (len(keys)):
        if values[index] < (score_mean -(score_mean *(1 -sigma))):
            print("Detect malicious client: ", keys[index])
            malicious.append(keys[index])
        else:
            print("Detected honest client")
            honest.append(keys[index])
    return honest,malicious

# print(verifier(Scores=Scores))

def verifier_wg(Scores, global_score):
    # sigma=0.85 #Hyperparameter to verify score
    malicious = []
    honest=[]
    for client_key in Scores:
        if Scores[client_key] < (global_score -(global_score *(1 -sigma))):
            print("Detect malicious client: ", client_key)
            malicious.append(client_key)
        else:
            print("Detected honest client")
            honest.append(client_key)
    if len(honest)==0:
       honest,malicious = verifier(Scores=Scores)
    return honest,malicious

# print(verifier(Scores=Scores))