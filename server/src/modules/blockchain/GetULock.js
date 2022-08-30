const fs = require("fs");

const BlockAPI = require('../block-api')

module.exports = async (ModelID) => {
    try {
        // let umodstr = fs.readFileSync(`./models/${ModelID}U.json`);
        let umodstr = await BlockAPI.Get(`${ModelID}U`);
        
        let umodel = JSON.parse(umodstr);
        console.log("Model Lock Status Get", umodel.ModelID);
        console.log("update Lock->", umodel.Lock);
        return umodel.Lock;
    } catch (error) {
        console.log("Error occured");
        console.error(error);
        return null;
    }
};