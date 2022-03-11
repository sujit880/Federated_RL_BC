const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        console.log("Passing Update score to blockchain", params[1]);
        let reply = Blockchain.UpdateScore(params);                
        if(!reply) throw "Problem in BC UpdateScore method"
        console.log(params[0],"Clients Scores are Updated.\n");
        return reply;
    } catch (error) {
        console.error(error);
        return null;
    }
};