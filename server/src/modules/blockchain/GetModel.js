const fs = require("fs");
const md5 = require("md5");
const BlockAPI = require("../block-api");

module.exports = async (params) => {
    try {
        client_key = params[2] + params[1];
        const key = md5(client_key);
        let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        await BlockAPI.Get(`${params[0]}`);

        let model = JSON.parse(model_str);

        console.log(model);

        console.log("Model_params_get", model.ModelID);

        let collection_str = fs.readFileSync(`./models/${model.ModelID}U.json`);
        await BlockAPI.Get(`${model.ModelID}U`);

        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration;
        console.log("check-B-GetModel1 after iteratio");
        // collected_params.AllClients.push("hello")
        console.log("All clients", collected_params.AllClients);
        if (!collected_params.AllClients.includes(key)) {
            collected_params.AllClients.push(key);
            collected_params.NClients = collected_params.AllClients.length;
            console.log("All clients after insertion", collected_params.AllClients);
        }

        let test_scores_str = fs.readFileSync(`./models/${collected_params.ModelID}_TS.json`);
        await BlockAPI.Get(`${collected_params.ModelID}_TS`);

        let test_scores = JSON.parse(test_scores_str);
        if (test_scores.Scores[key] === undefined) {
            test_scores.Scores[key] = test_scores.Base_Score;
            console.log("New score is given to client key", key, "\nScores:", test_scores.Scores);
        }
        console.log(
            "\nALL clients scores: ",
            Object.keys(test_scores.Scores),
            " -> ",
            Object.values(test_scores.Scores)
        );
        fs.writeFileSync(`./models/${test_scores.ModelID}_TS.json`, JSON.stringify(test_scores));
        await BlockAPI.Set(`${test_scores.ModelID}_TS`, JSON.stringify(test_scores));

        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        await BlockAPI.Set(`${collected_params.ModelID}U`, JSON.stringify(collected_params));

        console.log("Updated collected params file", collected_params.ModelID);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
