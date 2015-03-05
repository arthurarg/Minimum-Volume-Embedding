
function exec()
    %test(2, 1, [1]);
    visualization(2, 1, 1);

% allows to plot the digits as images
% verification purpose
function aff()
    X=importdata('digits9.mat');
    for i=1:155
        [p,img]=phi(X(i,:), 01);
        image( 12*img/max(p) )
        pause(0.01);
    end

% maps each digit list to a 27^2 image-vector
% also return a 27*27-matrix for visualization purpose
% a is a blur-parameter
function [p, img]=phi(digit, a)
    w=27;
    m=zeros(w^2,1);
    img=zeros(w,w);
    
    for i = 1:w;
        for j = 1:w;
            for k = 1:70;
                c=[digit(2*k-1), digit(2*k)];
                m(j+(i-1)*w,1) = m(j+(i-1)*w,1)+exp(-0.5*(norm(c-[i,j])^2)/a^2);
                img(i,j) = img(i,j)+exp(-0.5*(norm(c-[i,j])^2)/a^2);
            end
        end
    end
    p=m;

    
% allows to visualize data
% if type=1: polynomial kernel, degree=param
% else gaussian kernel, sigma=param
function visualization(bVal, type, param)
    
    targetd = 2;

    tol = 0.99;
    
    
    X=importdata('digits9.mat');
    
    % maps each digit to the image-vector
    x2=zeros(155,729);
    for i=1:155;
        [x2(i,:),img]=phi(X(i,:), 1);
    end
    x2=(x2')/max(max(x2));
    
    if type==1;
        A=(1+x2'*x2).^param;
    else
        A = calculateAffinityMatrix(x2, 2, param);
    end;
    
    G = convertAffinityToDistance(A);
    neighbors = calculateNeighborMatrix(G, bVal, 1);
    [Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);
    [Ykpca, origEigs] = kpca(A);
    
    disp('fid:')
    disp((eigVals(1)+eigVals(2))/sum(eigVals))
    disp((origEigs(1)+origEigs(2))/sum(origEigs))
    
    plotEmbedding(Y, neighbors, 'MVE embedding' ,36, x2)
    plotEmbedding(Ykpca, neighbors, 'KPCA embedding' ,37, x2)
    plotCompareEigSpectrums(origEigs, eigVals, 3);
    
    save results_q2.mat

% compute fidelity for p in params
% plot fidelity vs param
function test(bVal, type, params)
    targetd = 2;
    tol = 0.99;
    X=importdata('digits9.mat');
    res=[];
    
    x2=zeros(155,729);
    for i=1:155;
        [x2(i,:),img]=phi(X(i,:), 1);
    end
    x2=(x2')/max(max(x2));
    
    fid=[]
    
    for p=params;
        
        if type==1;
            A=(1+x2'*x2).^p;
        else
            A = calculateAffinityMatrix(x2, 2, p);
        end;
        
        G = convertAffinityToDistance(A);
        neighbors = calculateNeighborMatrix(G, bVal, 1);
        [Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);
        fid=[fid (eigVals(1)+eigVals(2))/sum(eigVals)]
        
    end
    disp(fid);
    figure(1);
    plot(params, fid);
    save results_q2.mat

    
    %
    % Auxiliary Functions
    %


% Calculates affinity matrix (Linear kernel, RBF)
function [A] = calculateAffinityMatrix(X, affType, sigma)
    [D,N] = size(X);
    disp(sprintf('Calculating Distance Matrix'));
    
    A = zeros(N, N);
    
    if affType == 1
        disp(sprintf('Using Linear Kernel'));
        A = X' * X; 
        %A = cov(X);
    elseif affType == 2
        disp(sprintf('Using RBF Kernel'));
        A = zeros(N, N);
        R1 = X' * X;
        Z = diag(R1);
        R2 = ones(N, 1) * Z';
        R3 = Z * ones(N, 1)';
        A  = exp((1/sigma) * R1 - (1/(2*sigma)) * R2 - (1/(2*sigma)) * R3);
    end



% Finds nearest neighbors in distance matrix G
function neighbors = calculateNeighborMatrix(G, bVal, type)

    N=length(G);
        
    if type==1
        disp(sprintf('Finding neighbors using K-nearest -- k=%d', bVal));
        [sorted,index] = sort(G);
        nearestNeighbors = index(2:(bVal+1),:);
        
        
        neighbors = zeros(N, N);
        for i=1:N
            for j=1:bVal
                neighbors(i, nearestNeighbors(j, i)) = 1;
                neighbors(nearestNeighbors(j, i), i) = 1;
            end
        end
        
    else
        disp(sprintf('Finding neighbors using B-matching -- b=%d', bVal));
        neighbors = permutationalBMatch(G, bVal);
        neighbors = neighbors .* (1 - eye(N));
    end


% Converts and affinity matrix to a distance matrix
function G = convertAffinityToDistance(A)
    N = size(A, 1);
    G = zeros(N, N);
    
    for i=1:N
        for j=1:N
            G(i, j) = A(i, i) - 2*A(i, j) + A(j, j);
        end
    end 

% Creates a plot comparing two eigenvalue spectrums
function plotCompareEigSpectrums(oEigs, mveEigs, figureNum);
    figure(figureNum);
    clf;
    subplot(2,1,1);
    bar(oEigs);
    title('Original Eigenvalues');
    subplot(2,1,2);
    bar(mveEigs);
    title('Eigenvalues after MVE');


% Plots a 2d embedding
function plotEmbedding(Y, neighbors, plotTitle, figureNum, x)
    figure(figureNum);
    clf;
    
    N = length(neighbors);
    
    scatter(Y(1,:),Y(2,:), 60,'filled');% axis equal;
    for i=1:N
        for j=1:N
            if neighbors(i, j) == 1
                line( [Y(1, i), Y(1, j)], [ Y(2, i), Y(2, j)], 'Color', [0, 0, 1], 'LineWidth', 1);
            end
        end
    end
    s1=(max(Y(1,:))-min(Y(1,:)))/30;
    s2=(max(Y(2,:))-min(Y(2,:)))/30;
    
    for i=1:155;
        hold on;
        imagesc([Y(1,i), Y(1,i)+s1], [Y(2,i), Y(2,i)+s2], reshape(x(:,i),27, 27) );
    end
    title(plotTitle);
    drawnow; 
    %axis off;
    

% Performs kernel principal component analysis
function [Y, eigV] = kpca(A);

    N = length(A);
    
    K = A - repmat(sum(A)/N, N, 1) - repmat((sum(A)/N)', 1, N) + sum(sum(A))/(N^2); K = (K + K')/2;
    [V, D]=eig(K);
    D0 = diag(D);
    V = V * sqrt(D);
    Y=(V(:,end:-1:1))';
    eigV=D0(end:-1:1);
    
    [eigV, IDX] = sort(eigV, 'descend');
    Y = Y(IDX, :);
    




%A = calculateAffinityMatrix(X, 1, 5);

