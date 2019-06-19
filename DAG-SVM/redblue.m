function c = redblue
% redblue gives a colormap within [cl(1) cl(2)] from
%   Blue [0 0 1] to White [0 0 0] to Red [1 0 0]

    m = size(get(gcf,'colormap'),1);        % Get size of colormap
    cl = caxis;                             % Get colormap limits
    m1 = round(-m*cl(1)/diff(cl));          % Get ratio of (-1), blue
    m2 = m - m1;                            %  the rest are (+1), red
    up = (0:m1-1)'/max(m1-1,1);             % up = [0,...,1] 
    dn = (m2-1:-1:0)'/max(m2-1,1);          % dn = [1,...,0]
    r = [up; ones(m2,1)];                   % red vector
    g = [up; dn];                           % green vector
    b = [ones(m1,1); dn];                   % blue vector
    c = [r g b];
end
