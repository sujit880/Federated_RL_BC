const fs = require("fs");
const ShardDB = require("../db");
const BlockAPI = require('../block-api')

module.exports = async (params) => {
    try{
        // let collection_str = fs.readFileSync(`./model/${params[0]}U.json`);
        // let collected_params = JSON.parse(collection_str);

        let score_str = fs.readFileSync(`./models/${params[0]}_TS.json`);
        await BlockAPI.Get(`${params[0]}_TS`);
        
        test_scores = JSON.parse(score_str);
        score = JSON.parse(params[1]);

        let keys = Object.keys(score);
        console.log("\n********Check delete after check********\nScores: ",score, "\nKeys: ", keys);
        for (let i=0; i<keys.length; i++){
            console.log("\n Score updating for client key: ", keys[i]);
            if ( test_scores.Scores[keys[i]] === undefined) throw "Error In Clients key security breach";
            test_scores.Scores[keys[i]] += score[keys[i]];
        }
        // Storing Score into file
        fs.writeFileSync(`./models/${test_scores.ModelID}_TS.json`, JSON.stringify(test_scores));
        await BlockAPI.Set(`${test_scores.ModelID}_TS`, JSON.stringify(test_scores));
        
        console.log("\nUpdated Test scores file..\n", test_scores.ModelID);
        return test_scores;
    }catch (error) {
        console.error(error);
        return null;
    }
};