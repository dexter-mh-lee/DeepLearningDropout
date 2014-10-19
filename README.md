Three types of layers: 
	- C: convolutional layer (matrix map)
	- MP: max-pooling layer (matrix map)
	- F: fully connected layer (vector map)
	- O: output layer

Convolutional Layers:
	- Scale: scale (size of patch)
	- Number of output maps: outputMap
	- Shared weights: k
	- Bias: b

Max-pooling layer
	- Scale: scale (size of patch)
	- Max-coordinate matrix: k (1 if max, 0 if not)

Fully connected layer (dimension and number of feature maps stay the same)
	- Weight matrix: w
	- Bias: b

Output layer (dimension equal to the dimension of output label)
	- Weight matrix: w
	- Bias: b

Common parameters
	- Result: a
	- Delta: d