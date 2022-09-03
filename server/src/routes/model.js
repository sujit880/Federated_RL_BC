const express = require("express");
const ModelContract = require("../modules/model");

const router = new express.Router();

router.post("/api/model/clear/:id", async (req, res) => {
    try {
        console.log("got clear request");
        let data = await ModelContract.ClearData([req.params.id]);
        if (data == null) throw "Error In Parent Methods.";

        res.status(200).send({ LockStatus: data });
    } catch (error) {
        console.error(error);
        res.status(404).send({ message: "Model Clear Status Error!" });
    }
});

router.get("/api/model/getLock/:id", async (req, res) => {
    try {
        // console.log("Socket: ", req.socket.remoteAddress);
        console.log("got getlock request from client ip: ", req.ip);
        let data = await ModelContract.GetLock([req.params.id]);
        console.log("lock data->", data)
        if (data == null) throw "Error In Parent Methods.";

        res.status(200).send({ LockStatus: data });
    } catch (error) {
        console.log("exception occured in getlock");
        console.error(error);
        res.status(404).send({ message: "Model Lock Status Error!" });
    }
});


router.get("/api/model/getULock/:id", async (req, res) => {
    try {
        let data = await ModelContract.GetULock([req.params.id]);
        if (data == null) throw "Error In Parent Methods.";

        res.status(200).send({ LockStatus: data });
    } catch (error) {
        console.error(error);
        res.status(404).send({ message: "Model Lock Status Error!" });
    }
});

router.get("/api/model/get/:id", async (req, res) => {
    try {
        
        console.log("got get request from client ip: ",req.ip, "pid: ", req.body.data.pid);
        let data = await ModelContract.GetModel([req.params.id, req.body.data.pid, req.ip]);
        if (!data) throw "Error In Parent Methods.";

        res.status(200).send(data);
    } catch (error) {
        console.log("exception occured in fetch param");
        console.error(error);
        res.status(404).send({ message: "Model NOT found!" });
    } });

// Worker side communication
router.get("/api/model/getclientP/:id", async (req, res) => {
    try {
        let data = await ModelContract.GetClientParams([req.params.id]);
        if (!data) throw "Error In Parent Methods.";

        res.status(200).send(data);
    } catch (error) {
        console.error(error);
        res.status(404).send({ message: "Clients Params NOT found!" });
    } });

router.post("/api/model/set/", async (req, res) => {
    try {
        console.log("\n\n.......**************...........\n       Setting Global Model\n");
        let reply = await ModelContract.SetModel([req.body.data.id, req.body.data.model, req.body.data.learning_rate, req.body.data.pid, req.ip]);
        if (!reply) throw "Error In Parent Methods.";

        res.status(200).send(reply);
    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Model NOT Set!" });
    }
});

router.set("/api/model/set_complete/:id", async (req, res) => {
    try {
        
        console.log("got set complete request from client ip: ",req.ip, "pid: ", req.body.data.pid);
        await ModelContract.SetComplete([req.params.id, req.body.data.pid, req.ip]);

        console.log("*******************Finishing model training**********************\n")
        if (!data) throw "Error In Parent Methods.";

        res.status(200).send({ message: "Set Complete Successful!" });
    } catch (error) {
        console.log("exception occured in fetch param");
        console.error(error);
        res.status(404).send({ message: "Model NOT found!" });
    } });

router.post("/api/model/collect/", async (req, res) => {
    try {
        console.log("Clients send local params.. \n Client ip: ", req.ip, "Client pid: ", req.body.data.pid);
        let reply = await ModelContract.PostParams([req.body.data.id, req.body.data.model, req.body.data.epochs, req.body.data.pid, req.ip]);
        if (!reply) throw "Error In Parent Methods.";

        res.status(200).send({ reply, message: "Client Params Successfully Collect." });
    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Params NOT Collect!" });
    }
});



router.post("/api/model/applyUpdate/:id", async (req, res) => {
    try {
        console.log("\n\nApplying Update")
        let reply = await ModelContract.ApplyModelUpdate([req.params.id, req.body.data.Params]);
        if (!reply) throw "Error In Parent Methods.";

        res.status(200).send({ reply, message: "Global Model Update Applied Successfully." });
    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Global Model Update NOT Applied!" });
    }
});

router.post("/api/model/updateScore/:id", async (req, res) => {
    try{
        console.log("\n\nGot UpdateScore Call.");
        let reply = await ModelContract.UpdateScore([req.params.id, req.body.data.Score]);
        if (!reply) throw "Error In Parents Method";
        console.log( "Scores For Clients Are Successfully Updated.\n");
        res.status(200).send({ reply, message: "Scores For Clients Are Successfully Updated."});
    }catch (error) {
        console.error(error);
        res.status(500).send({ message: "Score Update NOT Applied!" });
    }
} );


module.exports = router;
