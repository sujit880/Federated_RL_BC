const fs = require("fs");
const ShardDB = require("../db");
const BlockAPI = require('../block-api')

module.exports = async (params) => {
    try {
        model = {
            ModelID: params[0],
            ModelParams: params[1],
            LearningRate: params[2],
            NPush: 50,
            Iteration: -1,
            ModelUpdateCount: -1,
            ModelReadLock: false,
            ModelUpdateLock: true,  // true means can't update
        };

        fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        await BlockAPI.Set(`${model.ModelID}`, JSON.stringify(model));
        
        console.log("Model Params Set", model.ModelID);
        // client_params_value = ShardDB.GetKeyValuePair(params[1]);
        // client_params_key = ShardDB.SetClientParamsPair([client_key , client_params_value]);
        collectedparams = {
            ModelID: params[0],            
            Iteration: -1,
            NClients:-1,
            AllParams: {},
            AllClients: [],
            Lock: false,  // false means active to collecting model params

        };
        // collectedparams.AllClients.push("hello")
        fs.writeFileSync(`./models/${collectedparams.ModelID}U.json`, JSON.stringify(collectedparams));
        await BlockAPI.Set(`${collectedparams.ModelID}U`, JSON.stringify(collectedparams));
        
        console.log("Created File for collected params", collectedparams.ModelID);

        test_scores = {
            ModelID: params[0],
            Base_Score: 200,
            Scores: {},
        }
        // collectedparams.AllClients.push("hello")
        fs.writeFileSync(`./models/${collectedparams.ModelID}_TS.json`, JSON.stringify(test_scores));
        await BlockAPI.Set(`${collectedparams.ModelID}_TS`, JSON.stringify(test_scores));
        
        console.log("Created File for Test scores", collectedparams.ModelID);
        
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
