const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");
const md5 = require("md5");

module.exports = (params) => {
    try {
        
        // if (!model.ModelReadLock ){
        //     model.ModelReadLock = true;
        //     fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        //     console.log(model.ModelID,"Model read lock updated", model.ModelReadLock);
        // } 
        // params.push(iteration);
        client_key = params[4]+params[3];
        const key = md5(client_key);
        params[1] = ShardDB.SetClientParamsPair([key, params[1] ]);
        const collected_params = Blockchain.CollectParams([params[0],params[1],params[2]]);  //Iteration not declared
        console.log(collected_params.ModelID, "Model params collected");
        return collected_params;
    }catch (error) {
        console.error(error);
        return null;
    }
}