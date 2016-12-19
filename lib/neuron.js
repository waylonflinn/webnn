/* fully connected (inner product) layer

	Weblas Functions Used

	* sgemm
 */
var weblas = require('weblas')
	Tensor = weblas.pipeline.Tensor,
	type = require('type-detect');


function Neuron(weights, bias, dropout){

	var N, K;

	if(type(bias) === "Tensor"){
		this.bias = bias;
		console.assert(this.bias.shape[0] === 1, "Bias Tensor must have first dimension of length one.");
		K = this.bias.shape[1];
	} else {
		K = bias.length;
		this.bias = new Tensor([1, K], bias);
	}

	// transpose weights are required
	if(type(weights) === "Tensor"){
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

	var T_in;
	// is the input a Tensor?
	if(type(input) === "Tensor"){
		// yes, just use it
		T_in = input;
	} else {
		// no, create a Tensor (uploads data to GPU)
		var N = this.weights.shape[1];
		T_in = new Tensor([1, N], input);
	}

	// do the matrix multiply
	//sgemm(n, m, k, scale, bBuffer, aBuffer, scale, bias);
	var output = weblas.pipeline.sgemm(this.scale, T_in, this.weights, this.scale, this.bias);

	// if we created the Tensor, delete it
	if(type(input) !== "Tensor"){
		T_in.delete();
	}

	return output;
};
