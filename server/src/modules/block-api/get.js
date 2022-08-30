const axios = require('axios')

module.exports = async (ModelID) => {
    const res = await axios.get(`http://${process.env.BLOCKCHAIN_API}/api/model/get/${ModelID}`);

    console.log('Get model from blockchain', ModelID, res)

    return JSON.stringify(res.JSON)
}