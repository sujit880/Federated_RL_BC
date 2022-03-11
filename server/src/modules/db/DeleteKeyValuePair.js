const fs = require("fs");

module.exports = (key) => {
    try {
        console.log("\n\n....*************************....\n Deleting Shard for GlobalModel: ", key);
        let value = fs.unlinkSync(`./models/shards/${key}.shard`);
        console.log("Deleted Model Shard", key);
        return value;
    } catch (error) {
        console.error(error);
        return null;
    }
};
