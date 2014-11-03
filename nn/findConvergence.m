function [ time, value ] = findConvergence( points, accuracy )

    %Finds the point at which a time series converges to it's last value,
    %with convergence defined as the point at which all later points are
    %within accuracy of last value
    %ex. if last value is 0.09, to within 5% 
    %convergence is 0.0855 to 0.0945.
	if nargin < 2
        accuracy = 0.05;
    end
    
    value = points(length(points));
    converged = abs(points - value) <= abs(value * accuracy);
    time = find(~converged, 1, 'last')+1;
end

