/* fully connected (inner product) layer

	sgemm
 */
var weblas = require('weblas');

var Tensor = weblas.pipeline.Tensor;

function Neuron(weights, bias, dropout){

	var K = bias.length;
	var N = (weights.length / K) | 0;

	if(K * N !== weights.length)
		throw new Error("Weights array must have length which is a multiple of bias array.");

	// transpose weights are required
	if(typeof weights === "Tensor"){
		this.weights = weights;
	} else {
		this.weights = new Tensor([N, K], weights).transpose();
	}

	if(typeof weights === "Tensor"){
		this.bias = bias;
	} else {
		this.bias = new Tensor([1, K], bias);
	}


	this.scale = (1.0 - dropout);
}

module.exports = Neuron;

Neuron.prototype.forward = function(input){

	var inputT;
	// is the input a Tensor?
	if(typeof input === "Tensor"){
		// yes, just use it
		inputT = input;
	} else {
		// no, create a Tensor (uploads data to GPU)
		var N = this.weights.shape[1];
		inputT = new Tensor([1, N], input);
	}

	// do the matrix multiply
	//sgemm(n, m, k, scale, bBuffer, aBuffer, scale, bias);
	var output = weblas.pipeline.sgemm(this.scale, inputT, this.weights, this.scale, this.bias);

	// if we created the Tensor, delete it
	if(typeof input !== "Tensor"){
		inputT.delete();
	}

	return output;
};
