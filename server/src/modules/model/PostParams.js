const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        let model_str = fs.readFileSync(`./model/${params[0]}.json`);
        let model = JSON.parse(model_str);
        iteration = model.Iteration;
        params.push(iteration);
        client_key = params[4]+params[3];
        params[1] = ShardDB.SetClientParamsPair([client_key, params[1] ]);
        const model = Blockchain.CollectParams([params[0],client_key,params[1],params[2],iteration]);
        console.log(model.ModelID, "Model params collected", model.ModelID);
    }catch (error) {
        console.error(error);
        return null;
    }
}