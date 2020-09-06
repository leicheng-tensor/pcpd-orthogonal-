%% Demo of Probabilistic Tensor CPD with Orthogonal Factors 
%% In order to run this file, the scr folder needs to be put into the same folder as this file.
close all;
clc;
clear all;

addpath(genpath(pwd)); % Add present working directory to the MATLAB search path
randn('state',0); % Fix the start of random seed
rand('state',0); % Fix the start of random seed

%% Generate a 12*12*12 complex-valued tensor with Rank 5. The third factor matrix is orthogonal
dim_list = [12,12,12]; %specify the dim of each tensor mode
num_of_orthogonal = 0; % 1 factor matrix is asssumed to be orthogonal 
tensor_rank = 5; % tensor rank is 5
% call function tensor_generation () to generate the desired complex_valued tensor
[X, factor_cell] = tensor_generation(dim_list, num_of_orthogonal, tensor_rank); 

%% Generate Gaussian Noises, Bernulli-Gaussian (BG) Outliers and Observation
SNR = 20;
W = noise_generation(SNR, X); % Generate Gaussian noise with SNR = 20 dB
E = BG_generation(size(X), 100, 0); % Generate BG with power 100, ratio 0 or 0.05
Y = X + W + E;

%% Run VBTCPDO
disp('Run VBTCPDO Algorithm');
disp('-------------------------------------');
learning_results = VBTCPDO(Y, num_of_orthogonal);
%% Evaluate the learning results
X_est = learning_results.X;
R_est = learning_results.R;
mse_X = norm(X_est(:)-X(:),2)^2;
disp('Evaluate tensor learning');
fprintf( 'mse_X = %g, R_est=%g \n',mse_X, R_est);
