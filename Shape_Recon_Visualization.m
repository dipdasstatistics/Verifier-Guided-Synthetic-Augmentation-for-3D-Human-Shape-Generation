close all;
clear;

%% Load the generated coefficients
generated_pc = readtable("generated_samples.csv");        % Change this to generated coefficients file
pc_coeffs = table2array(generated_pc);

%% Load PCA Bases
load("real_scores_6motions.mat",'Mu','eigenVects');

%% Parameters

n_samples = size(pc_coeffs, 1);        % total number of generated shapes
n_pc = size(pc_coeffs, 2);
ReconShapes = zeros([256,256,3,n_samples]);

mu = Mu;        % mean shape
V = eigenVects; 

for idx = 1:n_samples
    % Reconstruct
    shape_v = mu + V * pc_coeffs(idx,:)';

    % Reshape to 256 × 256 × 3
    shape = reshape(shape_v, 256, 256, 3);

    ReconShapes(:,:,:,idx) = shape;
end

%% Visualize multiple(single) shapes
visual_id = 7:120:n_samples;         % Plot every 50 shapes. Can use other numbers.
% visual_id = 1;           % Use this if visualizing one shape at each time

for i = visual_id
    shape = ReconShapes(:,:,:,i);
    
    figure(i);
    surface(squeeze(shape(:,:,1)), squeeze(shape(:,:,2)), squeeze(shape(:,:,3)), ...
    ones(256,256), 'EdgeColor','none', 'SpecularExponent', 100);

    axis equal off;
    shading interp;
    lighting phong;
    camlight headlight;

    title(sprintf('Reconstructed Shape - Sample %d', i));
end