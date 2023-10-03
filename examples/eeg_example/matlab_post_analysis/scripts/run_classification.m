
% ==============================================================================
%
%        Neural Flows- Classification (Alzheimer's Disease vs Healthy Controls)
%                      Author: Matteo Vinao Carl
%        Affiliation: Imperial College London, Grossman Lab
%                   Date: 29th September 2023
%        Cite: Implicit Gaussian process representation of vector fields 
%        over arbitrary latent manifolds (2023). arXiv:2309.16746

% ==============================================================================
%
%                                 File Requirements:
%       File Name: sx_FlowFields.mat
%       Description: Time resolved divergence and vorticity per subject.
%       

%       File Name: boundary_mask.mat
%       Description: mask for convex hull 
%
% ==============================================================================

clc, clear
addpath(genpath(pwd))
params = config_params();


%% compute brain nodes which differ between groups

%compile subject data
ClusterRawData = get_ClusterData(params);

% compute significnat node clusters
ClusterPermTest = get_PermData(ClusterRawData,params);

%% extract top brain regions from ground truth EEG. 
SVMClassify = get_brainSeeds(ClusterRawData,ClusterPermTest,params);

% binary classification
SVMResults = svmClassify(SVMClassify,params);

%% save 
outputfolder= strcat(params.paths.root_dir,'/mat');
if ~exist(outputfolder,'dir')
    mkdir(outputfolder)

end
cd(outputfolder)
save('SVMResults','SVMResults','-v7.3')


