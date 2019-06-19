function CL = DAGsvm(x,y,xt,yt,kw,C)
% DAGsvm(x,y,...) runs a multi-class SVM using the DAGSVM algorithm
%   by Platt et al. (2000) [1]. For K classes, we train K(K-1)/2
%   1-norm soft or hard margin SVM classifiers with the RBF kernel. 
%
%   Inputs:     x  = training data [N x F]; where
%                    F = no. of features
%                    N = no. of training samples
%               y  = labels [N x 1]; y \in [1,2,..K]
%                    K = no. of classes
%               xt = test data [T x F]; where
%                    T = no. of test samples
%               yt = labels of test data [T x 1]
%               kw = desired kernel width; applies to all classifiers
%                  = -1, if you want to estimate kw instead [2]
%               C  = box constraint; applies to all classifiers
%
%   Outputs:    1. List of estimated kernel widths (if kw = -1)
%               2. List of misclassified training data (if there are any)
%               3. Plot of all K*(K-1)/2 classifiers in a DAG
%               4. Plot (gscatter) of all training data
%
%   Note: We reuse all training data as test data for visualization
%         purposes only. In practice, test data must be separate.
%
%   Refs: [1] Platt et al. Large Margin DAGs for Multiclass
%             Classification, Advances in NIPS, 2000.
%         [2] Karatzoglou et al. Support Vector Machines in R,
%             Journal of Statistical Software, 15(9), 2006.
%         [3] Support Vector Machines, Cristianini & Shawe-Taylor, 2000

clc; fprintf('Welcome to DAG-SVM!\n');
if nargin == 0
    fprintf('[1] (3 classes) FISHER IRIS - PETALS\n');
    fprintf('[2] (4 classes) FAN W/ 4 ARMS\n');
    fprintf('[3] (6 classes) RANDOM CIRCLES\n');
    fprintf('[4] (5 classes) SOUTHEAST ASIAN MAP\n');
    fprintf('[5] (7 classes) RAINBOW\n');
    ch = input('Choose dataset: ');             % Let the user choose
    
    switch ch
        case 1 % Set 1: FISHER IRIS (Petals data only)
        load fisheriris meas species;
        x = meas(:,3:4);    % x = [length, width]
        y = zeros(length(species),1);
        y(strcmp(species,'setosa') == 1) = 1;
        y(strcmp(species,'versicolor') == 1) = 2;
        y(strcmp(species,'virginica') == 1) = 3;
        xt = x; yt = y;     % Let all training set = test set
        C = 10;             % Recommended box constraint
        kw = -1;            % Let us estimate kw
    
        case 2 % Set 2: FAN W/ 4 ARMS
        load fan x;
        y = x(:,3); x = x(:,1:2);
        xt = x; yt = y;     % Let all training set = test set
        C = Inf;            % Recommended box constraint
        kw = 10;            % Recommended kernel width
        
        case 3 % Set 3: RANDOM CIRCLES
        x = zeros(600,2); y = ones(length(x),1);
        for j = 1:6
            ind = (1:100) + 100*(j-1);
            y(ind) = j; rd = 2*(rand + 0.5);
            t = 2*pi*rand(1,100); 
            r = rd*rand(1,100);
            x(ind,1) = r.*cos(t) + 10*rand;
            x(ind,2) = r.*sin(t) + 10*rand;
        end
        xt = x; yt = y;     % Let all training set = test set
        C = 1;              % Recommended box constraint
        kw = -1;            % Let us estimate kw
        
        case 4 % Set 4: SOUTHEAST ASIAN MAP
        load SEasia x country;
        y = x(:,3); x = x(:,1:2); % x = [Vsg, Vsl]
        xt = x; yt = y;     % Let all training set = test set
        C = 1e3;            % Recommended box constraint
        kw = 10;            % Recommended kernel width
        
        case 5 % Set 5: RAINBOW
        x = 5*(2*rand(500,2)-1); 
        y = ones(length(x),1);
        for j = 6:-1:1
            y(x(:,2) + 2*j - 7 > 0.5*(x(:,1)...
                + sin(2*x(:,1)))) = 8 - j;
        end
        xt = x; yt = y;     % Let all training set = test set
        C = Inf;            % Recommended box constraint
        kw = 0.5;           % Recommended kernel width
    end
end

%% SVM TRAINING FOR MULTI-CLASS CLASSIFICATION
%  See Platt et al. [1]

K = size(unique(y));                          % number of classes
CL = cell(K); c = 1;                            % classifiers [K x K]
for j = 1:(K-1)
    for k = K:-1:(j+1)
        xPos = x(y == j,:); xNeg = x(y == k,:); % (+) and (-) samples
        pN = size(xPos,1);  nN = size(xNeg,1);  % No. of samples
        Y = [ones(pN,1); -ones(nN,1)];          % Assign (+1) and (-1)
        KW = kw; X = [xPos; xNeg];
        if kw == -1
            KW = getKw(X);                      % Estimate kernel width
            fprintf('Estimated kw (%d vs %d): %.2f\n',j,k,KW)
        end
        CL{j,k} = binarySVM(X,Y,KW,C);          % Perform SVM
        %subplot(K-1,K-1,c);                     % K-1 by K-1 subplots
        %visCL(CL{j,k},j,k);                     % Plot the classifier
        c = c + 1; if k == j+1, c = c + j - 1; end
    end
end
set(gcf,'color','w');

%% SVM CLASSIFICATION ON TEST DATA, xt,yt
%  Check for misclassified data from training set

miss = 0;                                       % Count misclassified
for j = 1:size(xt,1)
    res = classify(xt(j,:),CL,K);               % Classify test data
    if res ~= yt(j)                             % Misclassified!
        miss = miss + 1;
        fprintf('x: (%.2f,%.2f); SVM: %d; Actual: %d\n',...
            xt(j,1),xt(j,2),res,yt(j));
    end
end
fprintf('Misclassified: %d / %d\n',miss,length(yt));


% %% PLOT MAP OF DECISION BOUNDARIES
% %  Boundaries are inferred from a plot of classified points
% 
% x1 = linspace(min(x(:,1)),max(x(:,1)));         % Limits on x1 axis
% x2 = linspace(min(x(:,2)),max(x(:,2)));         % Limits on x2 axis
% [X,Y] = meshgrid(x1,x2);
% Z = zeros(size(X)); 
% for jX = 1:length(X)
%     for jY = 1:length(Y)
%         xi = [X(jX,jY) Y(jX,jY)];
%         Z(jX,jY) = classify(xi,CL,K);           % Classify (x1,x2)
%     end
% end
% 
% switch ch                                       % Plot all training data
%     case 1
%     subplot(224);
%     gscatter(X(:),Y(:),Z(:));
%     legend(gca,'off'); hold on;
%     gscatter(x(:,1), x(:,2), species);
%     xlabel('Petal length (cm)');
%     ylabel('Petal width (cm)');
%     axis tight;
%     
%     case 2
%     subplot(339);
%     gscatter(X(:),Y(:),Z(:));
%     legend(gca,'off'); hold on;
%     gscatter(x(:,1), x(:,2), y);
%     axis tight;
%     
%     case 3
%     subplot(5,5,[19:20, 24:25]);
%     gscatter(X(:),Y(:),Z(:));
%     legend(gca,'off'); hold on;
%     gscatter(x(:,1), x(:,2), y);
%     axis image;
%     
%     case 4
%     subplot(4,4,[11:12 15:16]);
%     gscatter(X(:),Y(:),Z(:));
%     legend(gca,'off'); hold on;
%     gscatter(x(:,1), x(:,2), country(y));
%     axis tight; 
%     
%     case 5
%     subplot(6,6,[22:24, 28:30, 34:36]);
%     gscatter(X(:),Y(:),Z(:));
%     legend(gca,'off'); hold on;
%     gscatter(x(:,1), x(:,2), y);
%     axis tight;
% end
% hold off;

end

%% FUNCTION FOR CLASSIFYING UNSEEN DATA, x
%  Traverses the DAG of K*(K-1)/2 classifiers

function F = classify(x,CL,K)
    r = 1; c = K;                               % Current row & column
    while r ~= c
        F = CL{r,c};                            % Pick the classifier
        ySVM = sign(func(x,F));                 % SVM classifier output
        if ySVM == 1, c = c - 1;                % Traverse DAG left
        else,         r = r + 1;                % Traverse DAG down
        end
    end
    F = r;                                      % Return result (r or c)
end

%% FUNCTIONS TO EVALUATE ANY UNSEEN DATA, x
%  [xT,y,a,b,kw,sv] are fixed after solving the QP.
%  f(x) = SUM_{i=sv}(y(i)*a(i)*K(x,xT(i))) + b; 

function F = func(x,F)                          % Version 1
    x = (x - F.xm)./F.xs;                       % Normalize
    KM = repmat(x,size(F.sv)) - F.xT(F.sv,:);   % d = (x - x')
    KM = exp(-sum(KM.^2,2)/F.kw);               % RBF: exp(-d^2/kw)
    F = sum(F.y(F.sv).*F.a(F.sv).*KM) + F.b;    % f(x)
end

function F = func2(x,xT,y,a,b,kw,sv)            % Version 2
    K = repmat(x,size(sv)) - xT(sv,:);          % d = (x - x')
    K = exp(-sum(K.^2,2)/kw);                   % RBF: exp(-d^2/kw)
    F = sum(y(sv).*a(sv).*K) + b;               % f(x)
end

%% FUNCTION FOR GETTING KERNEL WIDTH, Kw
%  See page 11 from Karatzoglou et al. (2006)

function F = getKw(x) 
    N = size(x,1); c = 1;
    dist = zeros(N*(N+1)/2,1);
    for j = 1:N
        for k = 1:j
            d = x(j,:) - x(k,:);
            dist(c) = d*d'; c = c + 1;          % || x - x' ||^2
        end
    end
    F = quantile(dist,0.8);                     % 0.8 quantile
end

%% FUNCTION FOR VISUALIZING CLASSIFIER, F(X)
%  Mesh plot covering x = [-3,3] & scatter plot of training data

function visCL(F,a,b)
    [X,Y] = meshgrid(-3:0.02:3,-3:0.02:3);      % Set plot bounds [-3,3]
    Z = zeros(size(X));                         % Initialize Z matrix
    for jX = 1:length(X)
        for jY = 1:length(Y)
            xi = [X(jX,jY) Y(jX,jY)];
            Z(jX,jY) = func2(xi,F.xT,F.y,...
                F.a,F.b,F.kw,F.sv);             % Solve for heights, Z
        end
    end
    mesh(X,Y,Z); hold on;                       % Plot the manifold
    colormap(gca,redblue);
    N = size(F.y); zp = zeros(N); 
    for j = 1:N
        zp(j) = func2(F.xT(j,:),F.xT,F.y,...
            F.a,F.b,F.kw,F.sv);                 % Project training data
    end
    scatter3(F.xT(F.y == -1,1),F.xT(F.y == -1,2),... 
        zp(F.y == -1),10,'filled');             % [-1] data as scatter
    scatter3(F.xT(F.y == 1,1),F.xT(F.y == 1,2),... 
        zp(F.y == 1),10,'filled');              % [+1] data as scatter
    title(['\color{red}' num2str(a) ...
        ' \color{black}vs \color{blue}' num2str(b)]);
    view(2); axis tight; hold off;              % Rotate: Top view
end

%% SVM TRAINING FOR BINARY CLASSIFICATION
%  See Eq. (7.1) or Proposition 6.12 in SVM book [3]

function F = binarySVM(x,y,kw,C)
    N = length(y);                              % Let N = no. of samples
    if isscalar(C), C = C*ones(N,1); end        % if C is scalar...
    xm = mean(x); xs = std(x);                  % Mean and Std. Dev.
    x = (x - xm(ones(N,1),:))./xs(ones(N,1),:); % Normalize data
    xT = x;                                     % Save training data set
    H = zeros(N);                               % For sum(ai*aj*yi*yj*K)
    f = -ones(N,1);                             % For sum(a)
    Aeq = y';                                   % For sum(a'*y) = 0
    Beq = 0;                                    % For sum(a'*y) = 0
    lb = zeros(N,1);                            % For 0 <= a
    ub = C;                                     % For a <= C
    for j = 1:N
        for k = 1:j
            d = x(j,:) - xT(k,:);
            H(j,k) = y(j)*y(k)*exp(-(d*d')/kw); % Create kernel matrix
            H(k,j) = H(j,k);                    %  using RBF kernel
        end
    end

    options = optimoptions('quadprog',...
        'Algorithm','interior-point-convex',... % Set solver options
        'Display','off');

    a = quadprog(H,f,[],[],...                  % Solve the QP (see the
        Aeq,Beq,lb,ub,[],options);              % definition of quadprog)

    tol = 1e-8;
    n1 = a < tol; n2 = (a > C - tol);
    if sum(n1) >= 1, a(n1) = 0; end             % Tolerate small errors
    if sum(n2) >= 1, a(n2) = C(n2); end         % Tolerate small errors
    sv = find(a > 0);                           % Select support vectors

    nb = find(a > 0 & a < C);                   % Points near boundary
    temp = zeros(size(nb));
    for j = 1:length(nb)
        temp(j) = 1/y(nb(j)) - func2(x(nb(j),:),xT,y,a,0,kw,sv);
    end
    b = mean(temp);                             % Estimate the bias, b
    F.xT = xT; F.sv = sv; F.kw = kw; 
    F.a = a; F.b = b; F.y = y;
    F.xm = xm; F.xs = xs;
end
