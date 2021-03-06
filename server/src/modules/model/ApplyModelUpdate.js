const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        // let bModel = Blockchain.GetModel([params[0]]);
        let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        let model = JSON.parse(model_str);
        const modelHistory = parseInt(process.env.NUM_MODEL_HISTORY);

        // Delete Old Param Shards
        ShardDB.DeleteKeyValuePair(model.ModelParams); // Delete old global params

        //
        model.ModelParams = ShardDB.SetKeyValuePair(params[1]);

        // Set Model History Length
        params.push(process.env.NUM_MODEL_HISTORY);

        const Umodel = Blockchain.ApplyModelUpdate(model);
        console.log("Assembled Model Update Set", params[0]);
        return Umodel;
    } catch (error) {
        console.error(error);
        return null;
    }
};
