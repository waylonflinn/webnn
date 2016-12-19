/* rectified linear unit layer

	sclamp
 */
var weblas = require('weblas'),
	Tensor = weblas.pipeline.Tensor,
	type = require('type-detect');


function ReLU(){
	this.max = 0;
}

module.exports = ReLU;

ReLU.prototype.forward = function(input){

	var T_in;
	// is the input a Tensor?
	if(type(input) === "Tensor"){
		// yes, just use it
		T_in = input;
	} else {
		// no, create a Tensor (uploads data to GPU)
		var N = input.length;
		T_in = new Tensor([1, N], input);
	}

	var output = weblas.pipeline.sclmp(0.0, null, T_in);

	// if we created the Tensor, delete it
	if(type(input) !== "Tensor"){
		T_in.delete();
	}

	return output;
}
