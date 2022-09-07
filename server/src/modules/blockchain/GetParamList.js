const fs = require("fs");
const BlockAPI = require("../block-api");

module.exports = async (ModelID) => {
    try {
        let all_params_str = fs.readFileSync(`./model/${ModelID}all_params.json`);
        await BlockAPI.Get(`${ModelID}all_params`);

        let all_params = JSON.parse(all_params_str);
        console.log("Get all params list", all_params.ModelID);
        return all_params;
    } catch (error) {
        console.error(error);
        return null;
    }
};
