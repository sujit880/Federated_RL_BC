const fs = require("fs");
const md5 = require("md5");
const BlockApi = require("../block-api");
const d = new Date();
const block_latency_log = [];

module.exports = async (params) => {
    try {
        // collectparams = {
        //     ModelID: params[0],
        //     ClientKey: params[1],
        //     ClientEpochs: params[2],
        //     UpdateIteration: params[3],
        // };
        // console.log("storing client params of client", collectparams.ClientKey);
        // fs.writeFileSync(`./all_params/${collectparams.ClientKey}.json`, JSON.stringify(collectparams));
        let client_key = params[3];
        let client_epochs = params[2];
        // let updated_iteration = params[3];

        // Read Global model....
        let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        await BlockApi.Get(`${params[0]}`);

        let model = JSON.parse(model_str);
        let iteration = model.Iteration;

        // Read Collection params file....
        let collection_str = fs.readFileSync(`./models/${params[0]}U.json`);
        await BlockApi.Get(`${params[0]}U`);

        let collected_params = JSON.parse(collection_str);

        // Updating Params Collection File to Log New Updates..
        if (!collected_params.AllClients.includes(client_key)) throw "Clients key Error! Security breach!!..";
        collected_params.AllParams[client_key] = params[1];

        total_collection = Object.keys(collected_params.AllParams).length;
        console.log("Total local collection: ", total_collection, "#Clients: ", collected_params.NClients);
        if (!model.ModelReadLock && total_collection >= collected_params.NClients) {
            model.ModelReadLock = true;
            fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));            
            let time1 = d.getMilliseconds();
            await BlockApi.Set(`${model.ModelID}`, JSON.stringify(model));
            console.log("Storing data in blockchain");
            let time2 = d.getMilliseconds();
            block_latency_log.push(time2-time1);
            fs.writeFileSync(`./models/${model.ModelID}latency_set.json`, JSON.stringify(block_latency_log));

            console.log(model.ModelID, "Model read lock updated", model.ModelReadLock);
        }
        // params.push(iteration);

        // Iteration checking and logging....
        iteration = collected_params.Iteration;
        if (iteration == -1) {
            collected_params.Iteration = 1;
        } else {
            collected_params.Iteration += 1;
        }

        console.log("Client Params collected");
        console.log("collected all local params: ", Object.values(collected_params.AllParams));

        if (total_collection == collected_params.NClients) {
            collected_params.Lock = true;
        }
        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        let time1 = d.getMilliseconds();
        await BlockApi.Set(`${collected_params.ModelID}U`, JSON.stringify(collected_params));
        console.log("Storing local params data in blockchain");
        let time2 = d.getMilliseconds();
        block_latency_log.push(time2-time1);
        fs.writeFileSync(`./models/${model.ModelID}latency_set.json`, JSON.stringify(block_latency_log));
        console.log("Created File for collected params", collected_params.ModelID);
        return collected_params;
    } catch (error) {
        console.error(error);
        return null;
    }
};
