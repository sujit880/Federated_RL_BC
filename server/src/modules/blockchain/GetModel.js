const fs = require("fs");
const md5 = require("md5");

module.exports = (params) => {
    try{
        
        client_key = params[1]+params[2];
        const key = md5(client_key);
        let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        let model = JSON.parse(model_str);
        
        console.log("Model_params_get", model.ModelID);

        let collection_str = fs.readFileSync(`./models/${model.ModelID}U.json`);
        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration;
        // collected_params.AllClients.push("hello")
        console.log("All clients", collected_params.AllClients);
        if (!collected_params.AllClients.includes(key)){
            collected_params.NClients = collected_params.NClients+1;
            collected_params.AllClients.push(key);
        }
        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        console.log("Updated collected params", collected_params.ModelID);
        return model;
    }
    catch (error){
        console.error(error);
        return null;
    }
};

