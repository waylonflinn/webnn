/* fully connected (inner product) layer

	sgemm
 */
var weblas = require('weblas');

function Neuron(weights, bias, dropout){

	var N = bias.length;
	var K = (weights.length / N) | 0;

	if(K * N !== weights.length)
		throw new Error("Weights array must have length which is a multiple of bias array.");

	// transpose weights are required
	this.weights = new weblas.pipeline.Tensor([N, K], weights);
	this.bias = new weblas.pipeline.Tensor([1, N], bias);
	this.dropout = dropout;
}

module.exports = Neuron;

Neuron.prototype.forward = function(input){

	var inputT;
	// fix input?
	if(typeof input === "Tensor"){
		inputT = input;
	} else {
		var K = this.weights.shape[1];
		inputT = new weblas.pipeline.Tensor([1, K], input);
	}

	var output = weblas.pipeline.sgemm(this.dropout, inputT, this.weights, 1.0, this.bias);

	if(typeof input !== "Tensor"){
		inputT.delete();
	}

	return output;
};
