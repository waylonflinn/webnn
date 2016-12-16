/* fully connected (inner product) layer

	sgemm
 */
var weblas = require('weblas');

var Tensor = weblas.pipeline.Tensor;

function Neuron(weights, bias, dropout){

	var K;
	var N;

	if(typeof bias === "Tensor"){
		this.bias = bias;
		console.assert(this.bias.shape[0] === 1, "Bias Tensor must have first dimension of length one.");
		K = this.bias.shape[1];
	} else {
		K = bias.length;
		this.bias = new Tensor([1, K], bias);
	}

	// transpose weights are required
	if(typeof weights === "Tensor"){
		console.assert(this.weights.shape[1] === K, "Weights and bias Tensors must have same second dimension.");
		this.weights = weights.transpose();
	} else {
		N = (weights.length / K) | 0;
		console.assert(K * N === weights.length, "Weights array must have length equal to a multiple of bias array length.");

		this.weights = new Tensor([N, K], weights).transpose();
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
