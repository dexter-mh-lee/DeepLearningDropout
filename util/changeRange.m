function X = changeRange(x)
%sets range of matrix x to [0, 1]

X = (x - min(x(:))) / (max(x(:)) - min(x(:)));