const fs = require("fs");

module.exports = (params) => {
    try{
        client_key = params[1]+params[2]
        let model_str = fs.readFileSync(`./model/${params[0]}.json`);
        let model = JSON.parse(model_str);
        
        console.log("Model_params_get", model.ModelID);

        let collection_str = fs.readFileSync(`./model/${model.ModelID+'U'}.json`);
        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration

        if (!collected_params.AllClints.include(collectparams.ClientKey))
            collected_params.NClients = collected_params.NClients+1;
            collected_params.AllClints.push(client_key);

        return model;
    }
    catch (error){
        console.error(error);
        return null;
    }
};

