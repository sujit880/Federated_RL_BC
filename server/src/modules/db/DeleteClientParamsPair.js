const fs = require("fs");

module.exports = (key) => {
    try {
        let value = fs.unlinkSync(`./all_params/${key}.shard`);
        console.log("\n\nDeleted Client Model Shard", key);
        return value;
    } catch (error) {
        console.error(error);
        return null;
    }
};
