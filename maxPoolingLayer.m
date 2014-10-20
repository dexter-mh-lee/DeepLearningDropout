function [z, k] = maxPoolingLayer(a, mapSize, scale)
	%This may be vectorized further please check
% 	z = zeros([mapSize(1:2), size(a,3)]); % holder for the result matrix
% 	k = zeros(size(a)); % holder for the transition matrix
% 	for i = 1:mapSize(1)
% 		for j = 1:mapSize(2)
% 			% Divide the whole matrix into patches and apply maxPooling
% 			[z(i,j,:)] ...
% 				= maxPooling(a(scale * (i - 1) + 1:scale * i, scale * (j - 1) + 1:scale * j, :));
% 		end
%     end
    z = arrayfun(@(i) ...
            arrayfun(@(j) ...
                maxPooling(a(scale * (i - 1) + 1:scale * i, scale * (j - 1) + 1:scale * j, :)), ...
                1:mapSize(2), 'UniformOutput', false), ...
            1:mapSize(1), 'UniformOutput', false);
    z = cell2mat(cellfun(@(maxArray) transpose3d(cell2mat(maxArray)), z, 'UniformOutput', false));
    z = transpose3d(z);
    k = expand(z, [scale, scale, 1]) == a;
end