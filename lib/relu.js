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
	// is the input an Array?
	if(type(input) === "Tensor"){
		// no, assume it's a Tensor
		T_in = input;
	} else {
		// yes, create a Tensor (uploads data to GPU)
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
