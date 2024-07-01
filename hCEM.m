%% hCEM_demo
%
% This demo illustrates the Hierarchical Suppression Method for
% Hyperspectral Target Detection in the paper
%
% Zhengxia Zou and Zhenwei Shi. Hierarchical Suppression Method for 
% Hyperspectral Target Detection. IEEE Transactions on Geoscience and 
% Remote Sensing. Article in Press.
% 
% Synthetic data:
%      size(X) = [224, 4096]                % spectral data with 224 bands
                                            % and 4096 pixels
%      size(groundtruth) = [64, 64]         % true target map
%      size(d) = [224, 1]                   % target spectrum
%
% ------------------------------------------------------------------------
%
% Please contact Zhengxia Zou (zhengxiazou@buaa.edu.cn) to report bugs or
% provide suggestions and discussions for the codes. 
%
% ------------------------------------------------------------------------
% Author:
%      Zhengxia Zou (zhengxiazou@buaa.edu.cn) 
%      Zhenwei Shi (shizhenwei@buaa.edu.cn)
% Date: July, 2015
% Version: 1.0
% ------------------------------------------------------------------------

clear all; close all; clc;

%% load synthetic data
display('laoding the synthetic data...');
load('synthetic_data.mat');
N = size(X,2); % pixel number
D = size(X,1); % band number
imgH = size(groundtruth,1); % image height
imgW = size(groundtruth,2); % image width
% add 30 dB Gaussian white noise
SNR = 30; 
for i = 1:size(X,2)
       X(:,i) = awgn(X(:,i), SNR);
end
% show groundtruth
figure; subplot(121); imshow(groundtruth); 
title('groundtruth'); hold on;


%% parameter settings
% To obtain the optimal performances, the parameters lambda and epsilon 
% should be optimized for each hyperspectral image.
lambda = 200;
epsilon = 1e-6;

%% hCEM algorithm
% initialization 
Weight = ones(1,N);
y_old = ones(1,N);
max_it = 100;
Energy = [];

% Hierarchical filtering process
display('hierarchical filtering...');
for T = 1:max_it
     
     for pxlID = 1:N
         X(:,pxlID) = X(:,pxlID).*Weight(pxlID);
     end
     R = X*X'/N;
     
     % To inrease stability, a small diagnose matrix is added 
     % before the matrix inverse process.
     w = inv(R+0.0001*eye(D)) * d / (d'*inv(R+0.0001*eye(D)) *d);
   
     y = w' * X;
     Weight = 1 - 2.71828.^(-lambda*y);
     Weight(Weight<0) = 0;
     
     res = norm(y_old)^2/N - norm(y)^2/N;
     fprintf('ITERATION: %5d, RES: %.5u \n', T, res);
     Energy = [Energy, norm(y)^2/N];
     y_old = y;
     
     % stop criterion:
     if (abs(res)<epsilon)
         break;
     end
     
     % display the detection results of each layer 
     fileName = sprintf('hCEM-Result-Layer-%d',T);
     hCEMMap = reshape(mat2gray(y),[imgH,imgW]);
     subplot(122); imshow(hCEMMap); 
     title(fileName); hold on;
     pause(0.1);
     
     % write the detection results of each layer to the current folder
     imwrite(mat2gray(hCEMMap), strcat(fileName,'.tif'));
end

%% plot output energy curve
figure; plot(Energy);
xlabel('layers'); 
ylabel('energy');
title('output energy on diffrent layers');

%% plot ROC curves
display('ploting ROC curves...');
outputs = mat2gray(y);
targets = reshape(groundtruth, 1, N);
[FPR,TPR] = myPlotROC(targets, outputs);

figure;
semilogx(FPR,TPR);
xlabel('false alarm rate'); 
ylabel('probability of detection');
title('ROC curves of hCEM algorithm');
hold on;
display('done.');

