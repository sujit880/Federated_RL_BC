const fs = require("fs");
const Blockchain = require("../blockchain");
module.exports = (params) => {
    try {
        // params[1] = ShardDB.SetKeyValuePair(params[1]);
        client_key = params[1]+params[2];
        Blockchain.SetComplete(params[0]);
        console.log("Got finish training request from client: ", client_key)
    } catch (error) {
        console.error(error);
        return null;
    }
};