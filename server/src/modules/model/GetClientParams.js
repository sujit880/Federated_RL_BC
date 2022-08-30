const fs = require("fs");
const Blockchain = require("../blockchain");
const SharDb = require("../db");
const BlockAPI = require('../block-api')

module.exports = async (params) => {
    try{
        console.log("Got Get cliemt params call.  :", params[0]);
        // let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        let model_str = await BlockAPI.Get(`${params[0]}`);
        
        let bModel = JSON.parse(model_str);
        
        let all_params = await Blockchain.GetClientParams(bModel.ModelID);
        if (all_params !==null){
            bModel.ModelReadLock = true;
        }
        // fs.writeFileSync(`./models/${bModel.ModelID}.json`, JSON.stringify(bModel));
        await BlockAPI.Set(`${bModel.ModelID}`, JSON.stringify(bModel));
        
        console.log("Model Lock status update..", bModel.ModelID);
        data = {
            clients: all_params,
            global_params: bModel.ModelParams,
        } ;
        return data ;
    }catch (error) {
        console.error(error);
        return null;
    }
};