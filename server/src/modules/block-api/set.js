const axios = require("axios");
const delay = (ms) => new Promise((res) => setTimeout(res, ms));

module.exports = async (ModelID, ModelJSON) => {
    const res = await axios.post(`http://${process.env.BLOCKCHAIN_API}/api/model/set/${ModelID}`, {
        data: ModelJSON,
    });

    // await delay(1500);

    console.log("Set model on blockchain", ModelID);
};
