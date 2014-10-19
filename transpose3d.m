function result = transpose3d(mat)
	result = arrayfun(@(k) mat(:,:,k)', 1:size(mat, 3), 'UniformOutput', false);
	result = cat(3,result{:});
end