const fs = require("fs");
const Blockchain = require("../blockchain");

module.exports = async (params) => {
    try {
        let bModelLock = await Blockchain.GetLockStatus(params);

        console.log("Assembled Model Lock Status", params[0]);
        return bModelLock;
    } catch (error) {
        console.error(error);
        return null;
    }
};
