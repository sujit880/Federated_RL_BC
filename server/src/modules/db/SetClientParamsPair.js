const fs = require("fs");
const md5 = require("md5");

module.exports = (params) => {
    try {
        const key = params[0];
        fs.writeFileSync(`./models/shards/${key}.shard`, params[1]);
        console.log("Model Shard Set", key);
        return key;
    } catch (error) {
        console.error(error);
        return null;
    }
};