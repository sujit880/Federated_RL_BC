const axios = require("axios");
const delay = (ms) => new Promise((res) => setTimeout(res, ms));

module.exports = async (ModelID) => {
    const res = await axios.get(`http://${process.env.BLOCKCHAIN_API}/api/model/get/${ModelID}`);
    // await delay(100);
    // console.log("Get model from blockchain", ModelID, res.data);
    // return res.data.JSON;
};
