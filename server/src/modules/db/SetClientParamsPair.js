const fs = require("fs");
// const md5 = require("md5");

const BlockAPI = require("../block-api");

module.exports = async (params) => {
    try {
        const key = params[0];
        console.log("store client params with key:->", key);
        fs.writeFileSync(`./all_params/${key}.shard`, params[1]);
        await await BlockAPI.Set(`${key}`, params[1]);

        console.log("Model Shard Set", key);
        return key;
    } catch (error) {
        console.error(error);
        return null;
    }
};
