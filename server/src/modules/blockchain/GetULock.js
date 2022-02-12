const fs = require("fs");

module.exports = (ModelID) => {
    try {
        let umodstr = fs.readFileSync(`./models/${ModelID+'U'}.json`);
        let umodel = JSON.parse(umodstr);
        console.log("Model Lock Status Get", model.ModelID);
        return umodel.Lock;
    } catch (error) {
        console.error(error);
        return null;
    }
};