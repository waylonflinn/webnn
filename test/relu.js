var ReLU = require('../lib/relu.js'),
//	loader = require('floader'), // browserify aware file loader (xhr in browser)
	weblas = require('weblas'),
	tape = require('tape');

weblas.test = require('weblas/lib/test');

var RTOL = 1e-05,
	ATOL = 1e-07;

tape("relu: 1x4", function(t){
	t.plan(1);

	var input = new Float32Array([1.0, 2.0, -1.0, 3.0, -5.0]);
	var expected = new Float32Array([1.0, 2.0, 0.0, 3.0, 0.0]);

	var relu = new ReLU();

	try{
		var result = relu.forward(input).transfer();
	}
	catch(ex){
		t.error(ex);
		return;
	}

	weblas.test.assert.allclose(t, result, expected, null, RTOL, ATOL);

});

var dataDirectory = 'test/data/relu/',
	testFile = 'small.json';

var matrixFiles = ['in.arr', 'out.arr'];

function generateTestCase(prefix, n, k){
	return function(t){

		var expected; // typed arrays

		// directory containing matrix data files for current test
		var testDirectory = dataDirectory + prefix + '/';

		// load matrices from files
		weblas.test.load(testDirectory, matrixFiles, function(err, matrices){

			if(err){
				t.skip("Unable to load files: " + err.message);
				t.end();

				return;
			}

			t.plan(1);

			// matrices is an array which matches matrixFiles
			var input = matrices[0],
				expected = matrices[1];

			if(!(input && input.length && input.length == k * n &&
				expected && expected.length && expected.length == k * n)){

				throw new Error("malformed data");
			}

			var relu = new ReLU();
			var inputT = new weblas.pipeline.Tensor([n, k], input);

			try{
				var result = relu.forward(inputT).transfer();
			}
			catch(ex){
				t.assert(false, ex);
				return;
			}

			weblas.test.assert.allclose(t, result, expected, null, RTOL, ATOL);
		});
	};
}

directory = "0001";

// in [1,4096]
// out [1,999]
var test = {
	N : 55,
	K : 5280
}

var testName = "relu: " + test.N + " x " + test.K;

tape(testName, generateTestCase(directory, test.N, test.K));
