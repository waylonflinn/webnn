/* max pooling layer

	sdwns
 */
var weblas = require('weblas'),
	Tensor = weblas.pipeline.Tensor,
	type = require('type-detect');

function Pool(size, stride){

	this.stride = stride;
	this.size = size;
}

module.exports = Pool;

Pool.prototype.forward = function(input, M, N, channels){

	var T_in;
	// is the input a Tensor?
	if(type(input) === "Tensor"){
		// yes, use or reshape it
		if(input.shape[0] !== M || input.shape[1] !== N * channels) {
			T_in = input.reshape([M, N * channels]);
		} else {
			T_in = input;
		}
	} else {
		// no, create a Tensor (uploads data to GPU)
		T_in = new weblas.pipeline.Tensor([M, N * channels], input);
	}

	//t3 = weblas.pipeline.sdwns(channels, factor, stride, input._tensor);
	var output = weblas.pipeline.sdwns(channels, this.size, this.stride, T_in);
	//var output = weblas.pipeline.sdwns(0.0, null, T_in);

	// if we created the Tensor, delete it
	if(type(input) !== "Tensor"){
		T_in.delete();
	}

	return output;
}
