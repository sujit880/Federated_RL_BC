const fs = require("fs");
const Blockchain = require("../blockchain");

module.exports = async (params) => {
    try {
        console.log("got getLock request");
        let ModelLock = await Blockchain.GetLock(params);

        console.log("Got Model Lock data", params[0], ModelLock);
        return ModelLock;
    } catch (error) {
        console.error(error);
        return null;
    }
};