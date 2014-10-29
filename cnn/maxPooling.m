function [maxValue] = maxPooling(matrix)
	% maxValue - vector of maximum values
	% indexMatrix - 3d matrix with same dimensions as the input, 1 if maximum 0 if not
	maxValue = max(max(matrix));
	%indexMatrix = arrayfun(@(i) matrix(:,:,i) == maxValue(i), 1:size(matrix,3), 'UniformOutput', false);
	%indexMatrix = cat(3, indexMatrix{:});
end