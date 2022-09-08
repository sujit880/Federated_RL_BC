const fs = require("fs");

module.exports = (ModelID) => {
    try {
        let modstr = fs.readFileSync(`./models/${ModelID}.json`);
        let model = JSON.parse(modstr);
        let umodstr = fs.readFileSync(`./models/${ModelID}U.json`);
        let umodel = JSON.parse(umodstr);
        console.log("Model Lock Status Get", model.ModelID,"lock info", model.ModelReadLock, "Iteration", model.Iteration);
        model.ModelComplete = true;
        umodel.ModelComplete = true;
        fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        console.log("Set Complete in global model", model.ModelID, model.ModelComplete);
        fs.writeFileSync(`./models/${umodel.ModelID}U.json`, JSON.stringify(umodel));
        console.log("Set Complete in clients updates", umodel.ModelID, umodel.ModelComplete);
    } catch (error) {
        console.error(error);
        return null;
    }
};