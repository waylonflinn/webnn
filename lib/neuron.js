/* fully connected (inner product) layer

	sgemm
 */
var weblas = require('weblas');

function Neuron(weights, bias, dropout){

	// transpose weights are required
	this.weights = new weblas.pipeline.Tensor([K, N], weights);
	this.bias = new weblas.pipeline.Tensor([1, N], bias);
	this.dropout = dropout;
}

Neuron.prototype.forward = function(input){

	var inputT;
	// fix input?
	if(typeof input === "Tensor"){
		inputT = input;
	} else {
		inputT = new weblas.pipeline.Tensor([]);
	}

	var output = weblas.pipeline.sgemm(this.droput, inputT, this.weights, this.dropout, this.bias);

	return output;
};
