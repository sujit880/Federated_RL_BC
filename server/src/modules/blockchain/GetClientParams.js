const fs = require("fs");
const SharDb = require("../db");
const BlockAPI = require('../block-api')

module.exports = async (ModelID) => {
    try{
        // let collection_str = fs.readFileSync(`./models/${ModelID}U.json`);
        let collection_str = await BlockAPI.Get(`${ModelID}U`);
        
        let collected_params = JSON.parse(collection_str);

        // let score_str = fs.readFileSync(`./models/${ModelID}_TS.json`);
        let score_str = await BlockAPI.Get(`${ModelID}_TS`);
        test_score = JSON.parse(score_str);

        let all_params_wscore={};

        let client_keys = Object.keys(collected_params.AllParams);
        console.log("\n*****Check delete after check****\nGet Clientparams-> Client keys:", client_keys,"\n All Params: ", collected_params.AllParams);
        for (let i=0; i<client_keys.length; i++){
            if(test_score.Scores[client_keys[i]] === undefined) throw "Error with clients key!!"            
            // params = SharDb.GetClientParamsPair(client_keys[i]);
            all_params_wscore[collected_params.AllParams[client_keys[i]]]=test_score.Scores[client_keys[i]];
        }
        console.log("Get all client params for aggregation", collected_params.ModelID);
        return all_params_wscore;
    }
    catch (error){
        console.error(error);
        return null;
    }
};

