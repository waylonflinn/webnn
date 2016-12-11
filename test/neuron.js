var Neuron = require('../lib/neuron.js'),
	aloader = require('arrayloader'),
	weblas = require('weblas'),
	tape = require('tape');

weblas.test = require('weblas/lib/test');

var RTOL = 1e-05,
	ATOL = 1e-07;

tape("neuron", function(t){
	t.plan(1);

	var weights = new Float32Array([1.0, 1.0, 1.0, 1.0]),
		bias = new Float32Array([3.5, 3.5, 3.5, 3.5]),
		dropout = 1.0;

	var input = new Float32Array([1.0]);
	var expected = new Float32Array([4.5, 4.5, 4.5, 4.5]);

	var neuron = new Neuron(weights, bias, dropout);

	try{
		result = neuron.forward(input).transfer();
	}
	catch(ex){
		t.error(ex);
		return;
	}

	weblas.test.assert.allclose(t, result, expected, null, RTOL, ATOL);

});
