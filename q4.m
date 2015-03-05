function exec()
    %best_k();
    %best_poly();
    %best_beta();
    visualization();
    
    
% allows to read num jpg images and load them in a matrix
% X contains the black&white vectors
% Xorig contains the original image as a vecor
function [X, Xorig]=readJpg(num)
    clc;
    addpath(genpath(pwd));
    addpath('/Users/Arthur/Documents/MATLAB/mve-05/AFLW/');
    folder = '/Users/Arthur/Documents/MATLAB/mve-05/AFLW';
    disp('OK')
    filePattern = fullfile(folder);
    jpegFiles = dir(filePattern);
    X=zeros(150^2,num);
    Xorig=zeros(3*150^2,num);
    for k = 1:num;
        
      baseFileName = jpegFiles(k+3).name;
      fullFileName = fullfile(folder, baseFileName);
      imageArray= imread(fullFileName);
      [X(:,k), Xorig(:,k)] = bw(imageArray);
    end

    
% for each image returns two vectors
% black and white image / original image
function [r, rorig]=bw(img)

if length(size(img))<3;
    temp=zeros(150, 150, 3);
    for j = 1:3;
        temp(:,:,j)=img;
    end;
    img=temp;
end
r=reshape( (img(:,:,1)+img(:,:,2)+img(:,:,3))/3, [], 1);
rorig=reshape(img, [], 1);


function visualization()
    bVal=2
    targetd = 2;

    tol = 0.99;
    [X, Xorig]=readJpg(300);
    save faces.mat X;
    
    A=(X'*X+1).^3;
    
    %A = calculateAffinityMatrix(X, 1, 5);
    G = convertAffinityToDistance(A);
    neighbors = calculateNeighborMatrix(G, bVal, 1);
    
    [Y, K, eigVals, mveScore] = mveB(A, neighbors, tol, targetd, 0.9);
    [Ymvu, K, mvuEigVals, mvuScore] = mvu(A, neighbors, targetd);
    [Ykpca, origEigs] = kpca(A);
    
    plotEmbedding(Y, neighbors, 'MVE embedding' ,36, Xorig)
    plotEmbedding(Ymvu, neighbors, 'MVU embedding' ,35, Xorig)
    plotEmbedding(Ykpca, neighbors, 'KPCA embedding' ,37, Xorig);
    plotCompareEigSpectrums(origEigs, eigVals, 3);
    plotCompareEigSpectrums(origEigs, mvuEigVals, 3);
    
    disp('fidelity');
    disp((eigVals(1)+eigVals(2))/sum(eigVals));
    disp((mvuEigVals(1)+mvuEigVals(2))/sum(mvuEigVals));
    disp((origEigs(1)+origEigs(2))/sum(origEigs));
    
    save results_q4_1.mat
    

% compute MVE-fidelity for several polynomial-kernels
function best_poly()
    targetd = 2;
    bVal=2;

    tol = 0.99;
    [X, Xorig]=readJpg(300);
    save faces.mat X;
    
    fid=[]
    
    ds=1:4;
    for d=ds;
        A=(X'*X+1).^d;
        G = convertAffinityToDistance(A);
        neighbors = calculateNeighborMatrix(G, bVal, 1);
        [Y, K, eigVals, mveScore] = mveB(A, neighbors, tol, targetd, 0.5);
        
        fid=[fid (eigVals(1)+eigVals(2))/sum(eigVals)];
    end
    save results_q4_2.mat
    figure(1);
    plot(ds, fid);
    
    save results_q4_2.mat

% compute MVE-fidelity with several beta
function best_beta()
    targetd = 2;
    bVal=2;

    tol = 0.99;
    [X, Xorig]=readJpg(300);
    save faces.mat X;
    
    fid=[]
    
    betas=0.8:0.01:0.95;
    for b=betas;
        A=(X'*X+1).^3;
        G = convertAffinityToDistance(A);
        neighbors = calculateNeighborMatrix(G, bVal, 1);
        [Y, K, eigVals, mveScore] = mveB(A, neighbors, tol, targetd, b);
        
        fid=[fid (eigVals(1)+eigVals(2))/sum(eigVals)];
    end
    save results_q4_2.mat
    figure(1);
    plot(betas, fid);
    
    save results_q4_2.mat

% compute MVE-fidelity for bVal=2..4
function best_k()
    targetd = 2;
    
    tol = 0.99;
    [X, Xorig]=readJpg(300);
    save faces.mat X;
    
    fid=[]
    
    bs=[2, 3, 4, 6, 8, 10];
    for bVal=bs;
        
        A = calculateAffinityMatrix(X, 1, 5);
        G = convertAffinityToDistance(A);
        neighbors = calculateNeighborMatrix(G, bVal, 1);
        [Y, K, eigVals, mveScore] = mveB(A, neighbors, tol, targetd, 0.5);
        fid=[fid (eigVals(1)+eigVals(2))/sum(eigVals)];
    end
    save results_q4_2.mat
    figure(1);
    plot(bs, fid);
    
    save results_q4_2.mat


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
    
    scatter(Y(1,:),Y(2,:), 60,'filled'); %axis equal;
    for i=1:N
        for j=1:N
            if neighbors(i, j) == 1
                line( [Y(1, i), Y(1, j)], [ Y(2, i), Y(2, j)], 'Color', [0, 0, 1], 'LineWidth', 1);
            end
        end
    end
    s1=(max(Y(1,:))-min(Y(1,:)))/40;
    s2=(max(Y(2,:))-min(Y(2,:)))/40;
    sz=size(Y);
    
    for i=1:sz(2);
        hold on;
        
        imagesc([Y(1,i), Y(1,i)+s1], [Y(2,i)+s2, Y(2,i)], reshape(uint8(x(:,i)),150, 150, 3) );
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
    
