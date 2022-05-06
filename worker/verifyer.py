import numpy as np

Scores = {}

Scores['key1']= 125
Scores['key2']= 170
Scores['key3']= 180
Scores['key4']= 190
Scores['key5']= 125
Scores['key6']= 125
Scores['key7']= 125

## Verify If Client Is malicious

def verifier(Scores):
    sigma=0.85 #Hyperparameter to verify score
    keys = list(Scores.keys())
    values = list(Scores.values())
    print("Keys: ", keys, "Values: ", values)
    score_mean = np.mean(np.array(values))
    print("Mean: ",score_mean)
    malicious = []
    honest=[]
    for index in range (len(keys)):
        if values[index] < score_mean*sigma:
            print("Detect malicious client: ", keys[index])
            malicious.append(keys[index])
        else:
            print("Detected honest client")
            honest.append(keys[index])
    return honest,malicious

print(verifier(Scores=Scores))