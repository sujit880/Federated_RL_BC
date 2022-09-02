const fs = require("fs");

const BlockAPI = require('../block-api')

module.exports = async (ModelID) => {
    try {
        let modstr = fs.readFileSync(`./models/${ModelID}.json`);
        await BlockAPI.Get(`${ModelID}`);
        
        let model = JSON.parse(modstr);
        console.log("Model Lock Status Get", model.ModelID,"lock info", model.ModelReadLock, "Iteration", model.Iteration);
        return model.ModelReadLock;
    } catch (error) {
        console.error(error);
        return null;
    }
};