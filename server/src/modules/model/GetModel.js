const fs = require("fs");
const Blockchain = require("../blockchain");
const SharDb = require("../db");

module.exports = (params) => {
    try{
        client_key = params[1]+params[2];
        params.push(client_key);
        let bModel = Blockchain.GetModel(params);
        bModel.ModelParams = SharDb.GetKeyValuePair(bModel.ModelParams);
        
        console.log("Assembled Model Params Get", bModel.ModelID);
        return bModel;
    }catch (error) {
        console.error(error);
        return null;
    }
};