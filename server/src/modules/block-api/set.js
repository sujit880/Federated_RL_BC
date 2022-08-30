const axios = require('axios')

module.exports = async (ModelID, ModelJSON) => {
    const res = await axios.post(`http://${process.env.BLOCKCHAIN_API}/api/model/set/${ModelID}`, {
        'data': ModelJSON
    });

    console.log('Set model on blockchain', ModelID)
}