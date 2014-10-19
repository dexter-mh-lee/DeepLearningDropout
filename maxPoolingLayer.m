function [z, k] = maxPoolingLayer(a, mapSize, scale)
	%This may be vectorized further please check
	z = zeros([mapSize(1:2), size(a,3)]); % holder for the result matrix
	k = zeros(size(a)); % holder for the transition matrix
	for i = 1:mapSize(1)
		for j = 1:mapSize(2)
			% Divide the whole matrix into patches and apply maxPooling
			[z(i,j,:), k(scale * (i - 1) + 1:scale * i, scale * (j - 1) + 1:scale * j, :)] ...
				= maxPooling(a(scale * (i - 1) + 1:scale * i, scale * (j - 1) + 1:scale * j, :));
		end
	end
end