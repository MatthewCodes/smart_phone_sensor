function F = SVMtrial(x,y,kw,C)
% SVMtrial(x,y,kw,C) runs a support vector machine for classifying
%   2D-data into 2 classes, denoted by 1 and -1. Training is done by
%   solving a quadratic programming problem using the interior-point 
%   method. The Gaussian radial basis function (RBF) is used as kernel.
%
%   Inputs:     x   = training data [N x 2]
%               y   = known classification, 1 (red) or -1 (blue)
%               kw  = desired kernel width of RBF
%               C   = regularization constant (scalar or vector);
%                     if isvector(C), C must have size [N x 1]
%
%   Output:     3D mesh plot of the learned manifold
%               F   = function f(x) for checking sign(f(x))
%
%   Note: If there are no inputs, the user can choose from the 
%         6 prepared datasets below.
%   Author: Karl Ezra Pilario, 
%   Created: 2016; Updated: 2019
%   Refs.:  Coursera - Machine Learning by Andrew Ng
%           Support Vector Machines, Cristianini & Shawe-Taylor, 2000

%% CHOOSE TRAINING SETS

clc;
fprintf('Welcome to SVM Trials!\n');
if nargin == 0
    fprintf('[1] TYPICAL\n');
    fprintf('[2] SADDLE\n');
    fprintf('[3] RANDOM\n');
    fprintf('[4] RANDOM, IN ELLIPSE W/ 1 OUTLIER\n');
    fprintf('[5] SPIRAL\n');
    fprintf('[6] IMBALANCED + OVERLAP\n')
    ch = input('Choose dataset: ');             % Let the user choose

    switch ch
        case 1
        % Set 1: TYPICAL
        kw = 0.8;   % Recommended RBF kernel width
        C = Inf;    % Recommended box constraint
        x = [4,5,2,2,4,9,7,8,8,9;
             7,8,2,5,5,2,1,1,5,4]';
        y = [1 1 1 1 1 -1 -1 -1 -1 -1]';

        case 2
        % Set 2: SADDLE
        kw = 1;     % Recommended RBF kernel width
        C = Inf;    % Recommended box constraint
        x = [4,4,2,3,8,9,7,7,5,4,6,5,8,9,6,7;
             4,6,6,3,2,3,2,0,1,0,1,2,7,8,7,5]';
        y = [1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1]';

        case 3
        % Set 3: RANDOM
        kw = 0.1;   % Recommended RBF kernel width
        C = 1;      % Recommended box constraint
        x = 10*rand(50,2);
        y = ones(50,1); y(1:25) = -1;

        case 4
        % Set 4: RANDOM, IN ELLIPSE W/ 1 OUTLIER
        kw = 0.25;  % Recommended RBF kernel width
        C = 10;     % Recommended box constraint
        x = 10*rand(150,2);
        y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
        y(y > 0) = 1; y(y ~= 1) = -1;
        outlr = randi(150);
        y(outlr) = -y(outlr); % Outlier (this is removable)

        case 5
        % Set 5: SPIRAL
        kw = 0.2;   % Recommended RBF kernel width
        C = Inf;    % Recommended box constraint
        x = importdata('myspiral.mat');
        y = x(:,3); x = x(:,1:2);   
        
        case 6
        % Set 6: IMBALANCED + OVERLAP
        kw = 0.5;   % Recommended RBF kernel width
        Cpos = 1;   % Recommended box constraint (red)
        Cneg = Inf; % Recommended box constraint (blue)
        x = importdata('imba.mat');
        y = x(:,3); x = x(:,1:2);
        C = zeros(size(y));
        C(y == 1) = Cpos; C(y == -1) = Cneg;
        % Remark: Try switching Cpos and Cneg.
    end
 
end

%% NORMALIZE DATA

N = length(y);                                  % Let N = no. of samples
xm = mean(x); xs = std(x);                      % Mean and Std. Dev.
x = (x - xm(ones(N,1),:))./xs(ones(N,1),:);     % Center and scale data
xT = x;                                         % Save training data set

%% SOLVE THE QUADRATIC PROGRAMMING PROBLEM
%  See Eq. (7.1) or Proposition 6.12 in SVM book

H = zeros(N);                                   % For sum(ai*aj*yi*yj*K)
f = -ones(N,1);                                 % For sum(a)
Aeq = y';                                       % For sum(a'*y) = 0
Beq = 0;                                        % For sum(a'*y) = 0
if isscalar(C), C = C*ones(N,1); end            % if C is scalar...
lb = zeros(N,1);                                % For 0 <= a
ub = C;                                         % For a <= C

for j = 1:N
    for k = 1:j
        d = x(j,:) - xT(k,:);
        H(j,k) = y(j)*y(k)*exp(-(d*d')/kw);     % Create kernel matrix
        H(k,j) = H(j,k);                        %  using RBF kernel
    end
end

options = optimoptions('quadprog',...
    'Algorithm','interior-point-convex',...     % Set solver options
    'Display','off');

a = quadprog(H,f,[],[],...                      % Solve the QP (see the
    Aeq,Beq,lb,ub,[],options);                  % definition of quadprog)

tol = 1e-8;
n1 = a < tol; n2 = (a > C - tol);
if sum(n1) >= 1, a(n1) = 0; end                 % Tolerate small errors
if sum(n2) >= 1, a(n2) = C(n2); end             % Tolerate small errors
sv = find(a > 0);                               % Select support vectors

fprintf('No. of Support Vectors: %d\n',length(sv));
fprintf('No. of Samples: %d\n',N);

%% ESTIMATE THE BIAS, b
%  b is chosen so that y(i)*f(x(i)) = 1 for i=nb

nb = find(a > 0 & a < C);                       % Points near boundary
temp = zeros(size(nb));
for j = 1:length(nb)
    temp(j) = 1/y(nb(j)) - func(x(nb(j),:),xT,y,a,0,kw,sv);
end
b = mean(temp);                                 % Estimate the bias, b

%% PLOT THE MANIFOLD IN 3D!!
%  Mesh plot covering x = [-3,3] & scatter plot of training data

[X,Y] = meshgrid(-3:0.01:3,-3:0.01:3);          % Set plot bounds [-3,3]
Z = zeros(size(X));                             % Initialize Z matrix
for jX = 1:length(X)
    for jY = 1:length(Y)
        xi = [X(jX,jY) Y(jX,jY)];
        Z(jX,jY) = func(xi,xT,y,a,b,kw,sv);     % Solve for heights, Z
    end
end
mesh(X,Y,Z); hold on;                           % Plot the manifold
colormap(redblue);
zp = zeros(size(y)); 
for j = 1:N
    zp(j) = func(xT(j,:),xT,y,a,b,kw,sv);       % Project training data
end
scatter3(xT(y == -1,1),xT(y == -1,2),...        % [-1] data as scatter
    zp(y == -1),'filled');
scatter3(xT(y == 1,1),xT(y == 1,2),...          % [+1] data as scatter
    zp(y == 1),'filled');

ch = input('Encircle support vectors? [0/1]: ');
if ch == 1
    plot3(xT(sv,1),xT(sv,2),zp(sv),'ko','MarkerSize',8);
end
hold off;
F.xT = xT; F.sv = sv; F.kw = kw; 
F.a = a; F.b = b; F.y = y;

%% FUNCTION TO EVALUATE ANY UNSEEN DATA, x
%  [xT,y,a,b,kw,sv] are fixed after solving the QP.
%  f(x) = SUM_{i=sv}(y(i)*a(i)*K(x,xT(i))) + b; 

    function F = func(x,xT,y,a,b,kw,sv)
        K = repmat(x,size(sv)) - xT(sv,:);      % d = (x - x')
        K = exp(-sum(K.^2,2)/kw);               % RBF: exp(-d^2/kw)
        F = sum(y(sv).*a(sv).*K) + b;           % f(x)
    end
end