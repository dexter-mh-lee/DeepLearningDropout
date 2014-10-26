function Y = makeY(y) 
%converts label vector to boolean matrix
%y is Nx1
N = length(y);
vals = unique(y);
Y = nan(N, length(vals));
for i=1:length(vals)
  if (iscellstr(vals))
    Y(:,i) = strcmp(y, vals(i));
  else
    Y(:,i) = y == vals(i);
  end
  
end