const fs = require("fs");

module.exports = (params) => {
    try {
        collectparams = {
            ModelID: params[0],
            ClientKey: params[1],
            ClientEpochs: params[2],
            UpdateIteration: params[3],
        };

        let collection_str = fs.readFileSync(`./model/${params[0]+'U'}.json`);
        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration
        if (iteration == -1)
            collected_params.Iteration = params[3];

        fs.writeFileSync(`./all_params/${collectparams.ClientKey}.json`, JSON.stringify(collectparams));
        collected_params.AllParams.push(collectparams.ClientKey);
        console.log("Client Params collected", collectparams.ClientKey);

        fs.writeFileSync(`./models/${collected_params.ModelID}.json`, JSON.stringify(collected_params));
        console.log("Created File for collected params", collected_params.ModelID);

        if (collected_params.AllParams.length== collected_params.NClients)
            collected_params.Lock = true;
        return collectparams;
    } catch (error) {
        console.error(error);
        return null;
    }
};
