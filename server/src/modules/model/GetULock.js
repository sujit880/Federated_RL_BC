const fs = require("fs");
const Blockchain = require("../blockchain");

module.exports = async (params) => {
    try {
        let UModelLock = await Blockchain.GetULock(params);

        console.log("Got UModel Lock data", params[0]);
        return UModelLock;
    } catch (error) {
        console.error(error);
        return null;
    }
};
