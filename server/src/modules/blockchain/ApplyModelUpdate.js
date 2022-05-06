const fs = require("fs");
const ShardDB = require("../db");

module.exports = (model) => {
    try {
        // model = params[0];
        // console.log("\n\n.....Checkeng model Delete after Check....\nModel: ", model);
        model.Iteration += 1; 
        model.ModelUpdateCount +=1;
        model.ModelReadLock = false;

        let collection_str = fs.readFileSync(`./models/${model.ModelID}U.json`);
        let collected_params = JSON.parse(collection_str);
        let remove_clients_list = [];
        let client_keys = Object.keys(collected_params.AllParams);
        for(let i=0; i<collected_params.AllClients.length;i++){
            if (client_keys.includes(collected_params.AllClients[i])){
                ShardDB.DeleteClientParamsPair(collected_params.AllClients[i]);
            }
            else{
                remove_clients_list.push(collected_params.AllClients[i]);
            }
        }

        // Removing inactive clients from all clients set.
        for (let i=0; i<remove_clients_list.length; i++){
            console.log("\n\n************************\nRemoving Clients");
            const index = collected_params.AllClients.indexOf(remove_clients_list[i]);
            if (index > -1) {
                collected_params.AllClients.splice(index, 1); // 2nd parameter means remove one item only
                console.log("\nClients Removed: ", remove_clients_list[i]);
            }
        }
        // Updated #clients for next iteration.
        collected_params.NClients = collected_params.AllClients.length;
        // while(collected_params.AllParams.length>0){
        //     ShardDB.DeleteClientParamsPair(collected_params.AllParams.pop());
        // }

        // Cleared All params of clients.
        collected_params.AllParams = {};
        collected_params.Lock = false;

        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        console.log("Created File for collected params", collected_params.ModelID);

        fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        console.log("Model Params Set", model.ModelID);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
