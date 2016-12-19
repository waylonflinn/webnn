var Pool = require('../lib/pool.js'),
//	loader = require('floader'), // browserify aware file loader (xhr in browser)
	weblas = require('weblas'),
	tape = require('tape');

weblas.test = require('weblas/lib/test');

var RTOL = 1e-05,
	ATOL = 1e-07;

tape("pool: 2 x 2 x 4", function(t){
	t.plan(1);

	var input = new Float32Array([	1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
								1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
		expected = new Float32Array([	1.0, 2.0, 1.0, 1.0 ]);

	var size = 2, stride = 2;
	var pool = new Pool(size, stride);
	var M = 2, N = 2, C = 4;

	try{
		// adapted from weblas.sdwns test
		//result = weblas.sdwns(2, 2, 4, 2, 2, X);
		var result = pool.forward(input, M, N, C).transfer();
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

function generateTestCase(prefix, M, N, C, size, stride){
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

			var len = (Math.floor((M - size) / stride) + 1) *
						(Math.floor((N - size) / stride) + 1) * C;

			if(!(input && input.length && input.length == M * N * C &&
				expected && expected.length && expected.length == len)){

				throw new Error("malformed data");
			}

			var pool = new Pool(size, stride);
			//var M = 2, N = 2, C = 4;

			try{
				//var result = relu.forward(inputT).transfer();

				// adapted from weblas.sdwns test
				//result = weblas.sdwns(2, 2, 4, 2, 2, X);
				var result = pool.forward(input, M, N, C).transfer();
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
	M :
	N : ,
	C : ,
	size : ,
	stride :
}

var testName = "pool: " + test.M + " x " + test.N + " x " + test.C;

tape(testName, generateTestCase(directory, test.M, test.N, test.C, test.size, test.stride));
