function dist = max_crosscovariance_dist_metric(x, Y)
% One minus the maximum value of the crosscorrelation of x and y
    % x is a 1-by-n vector
    % Y is a m-by-n matrix whose rows are to be compared with x
    % dist is m-by-1 vector

    m = size(Y, 1); 
    dist_func = @(xy) (1 - max(xcov(xy(1:end/2), xy(end/2+1:end), round(m/4)), [],'all'));
    dist = rowfun(dist_func, table([repmat(x, size(Y,1), 1), Y]));
    dist = dist{:,1};
end

