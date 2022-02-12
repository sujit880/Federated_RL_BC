const fs = require("fs");
const SharDb = require("../db");

module.exports = (params) => {
    try{
        let collection_str = fs.readFileSync(`./model/${params[0]}.json`);
        let collected_params = JSON.parse(collection_str);
        let all_params=[]
        for (let i=0; i<collected_params.AllParams.length; i++){
            collect_params = fs.readFileSync(`./all_params/${collected_params.AllClients[i]}.json`)
            params = SharDb.GetClientParamsPair(collect_params.ClientParams);
            all_params.push(params)
        }
        console.log("Get all client params for aggregation", collected_params.ModelID);
        return all_params;
    }
    catch (error){
        console.error(error);
        return null;
    }
};

