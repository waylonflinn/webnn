var Neuron = require('../lib/neuron.js'),
//	loader = require('floader'), // browserify aware file loader (xhr in browser)
	weblas = require('weblas'),
	tape = require('tape');

weblas.test = require('weblas/lib/test');

/*  run in a browser with testling

		browserify test/*.js | testling -x google-chrome

	on Ubuntu, requires

		sudo apt-get install xvfb
 */

var RTOL = 1e-05,
	ATOL = 1e-07;

tape("neuron 1x4", function(t){
	t.plan(1);

	var weights = new Float32Array([1.0, 1.0, 1.0, 1.0]),
		bias = new Float32Array([3.5, 3.5, 3.5, 3.5]),
		dropout = 0.0;

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

var dataDirectory = 'test/data/neuron/',
	testFile = 'small.json';

var matrixFiles = ['w.arr', 'b.arr', 'in.arr', 'out.arr'];

function generateTestCase(prefix, n, k, dropout){
	return function(t){

		var weights, bias, expected; // typed arrays

		// directory containing matrix data files for current test
		var testDirectory = dataDirectory + prefix + '/';

		//console.log(testDirectory);
		// load matrices from files
		weblas.test.load(testDirectory, matrixFiles, function(err, matrices){

			if(err){
				t.skip("Unable to load files: " + err.message);
				t.end();

				return;
			}

			t.plan(1);

			// matrices is an array which matches matrixFiles
			var weights = matrices[0],
				bias = matrices[1],
				input = matrices[2],
				expected = matrices[3];

			if(!(weights && weights.length && weights.length == n * k &&
				bias && bias.length && bias.length == 1 * k &&
				input && input.length && input.length == 1 * n &&
				expected && expected.length && expected.length == 1 * k)){

				throw new Error("malformed data");
			}

			//console.log(m + "x" + k + " times " + k + "x" + n);

			var neuron = new Neuron(weights, bias, dropout);

			try{
				result = neuron.forward(input).transfer();
				//result = weblas.sgemm(m, n, k, alpha, A, B, beta, null);
			}
			catch(ex){
				t.assert(false, ex);
				return;
			}

			weblas.test.assert.allclose(t, result, expected, null, RTOL, ATOL);
		});
	};
}

directory = "0001";//String("0000" + (i + 1)).slice(-4);

// w [4096,999]
// b [999]
// in [1,4096]
// out [1,999]
var test = {
	dropout : 0.0,
	N : 4096,
	K : 999
}
// k = dims[1]

var testName = "neuron: " + test.N + " x " + test.K;

tape(testName, generateTestCase(directory, test.N, test.K, test.dropout));
