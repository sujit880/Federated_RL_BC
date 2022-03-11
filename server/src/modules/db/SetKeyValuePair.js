const fs = require("fs");
const md5 = require("md5");

module.exports = (value) => {
    try {
        const key = md5(value);
        console.log("\n\n....*************************....\n Creating Shard for GlobalModel: ", key);
        fs.writeFileSync(`./models/shards/${key}.shard`, value);
        console.log("Model Shard Set", key);
        return key;
    } catch (error) {
        console.error(error);
        return null;
    }
};
