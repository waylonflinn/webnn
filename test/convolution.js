var Convolution = require('../lib/convolution.js'),
//	loader = require('floader'), // browserify aware file loader (xhr in browser)
	weblas = require('weblas'),
	tape = require('tape');

weblas.test = require('weblas/lib/test');

var RTOL = 1e-05,
	ATOL = 1e-12;
/*
tape("convolution: 2 x 2 x 4", function(t){
	t.plan(1);

	var weights = new Float32Array([1.0, 1.0, 1.0, 1.0]),
		bias = new Float32Array([3.5, 3.5, 3.5, 3.5]);

	var input = new Float32Array([	1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
								1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
		expected = new Float32Array([	1.0, 2.0, 1.0, 1.0 ]);

	var size = 2, count = 2, stride = 2, margin = 1;

	var conv = new Convolution(kernels, size, count, stride, bias, margin);
	var M = 2, N = 2, C = 4;

	try{
		// adapted from weblas.sdwns test
		//result = weblas.sdwns(2, 2, 4, 2, 2, X);
		var result = conv.forward(input, M, N, C).transfer();
	}
	catch(ex){
		t.error(ex);
		return;
	}

	weblas.test.assert.allclose(t, result, expected, null, RTOL, ATOL);

});
*/

var dataDirectory = 'test/data/convolution/',
	testFile = 'small.json';

var matrixFiles = ['k.arr', 'b.arr', 'in.arr', 'out.arr'];

function generateTestCase(prefix, M, N, C, size, count, stride, margin){
	return function(t){

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
			var kernels = matrices[0],
				bias = matrices[1],
				input = matrices[2],
				expected = matrices[3];

			// output of slokn
			w_lin = M + (2 * margin);
			h_lin = N + (2 * margin);

			// width and height of output
			w = Math.round(Math.ceil((w_lin - size) / stride) + 1);
			h = Math.round(Math.ceil((h_lin - size) / stride) + 1);

			var len =  w * h * count;
			if(!(input && input.length && input.length == M * N * C &&
				bias && bias.length && bias.length == count &&
				expected && expected.length && expected.length == len)){

				throw new Error("malformed data");
			}

			var conv = new Convolution(kernels, size, count, stride, bias, margin);

			//var M = 2, N = 2, C = 4;

			try{

				// adapted from weblas.sdwns test
				var result = conv.forward(input, M, N, C);

				result = result.transfer();
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

// in [27,1296]
// out [27,3456]

// in [27,1296]
// bias [128, 1]
// kernels [1200,128]
// out []
var test = {
	M : 27,
	N : 27,
	C : 48,
	size : 5,
	count : 128,
	stride : 1,
	margin : 2
}

var testName = "convolution: " + test.M + " x " + test.N + " x " + test.C;

tape(testName, generateTestCase(directory, test.M, test.N, test.C, test.size, test.count, test.stride, test.margin));
