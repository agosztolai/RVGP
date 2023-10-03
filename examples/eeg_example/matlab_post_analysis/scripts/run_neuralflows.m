
% ==============================================================================
%
%                                 Neural Flows
%                      Author: Matteo Vinao Carl
%        Affiliation: Imperial College London, Grossman Lab
%                   Date: 29th September 2023
%        Cite: Implicit Gaussian process representation of vector fields 
%        over arbitrary latent manifolds (2023). arXiv:2309.16746
% ==============================================================================
%
%                                 File Requirements:
%       File Name: obj_flows.mat
%       Description: EEG phase fields per subject (spline interpolated and
%       ground truth EEG).
%
%       File Name: results.mat
%       Description: RVGP reconstructed phase vector field per subject.

%       File Name: rsf_chanlocs.mat
%       Description: eeglab chanlocs structure contains 256 electrode positions.
%
%       File Name: rsf_participants.mat
%       Description: behavioral data.
%
%       File Name: B_rsf.mat
%       Description: CNEM file for calculating spatio-temporal derivatives .
%
% ==============================================================================

clc, clear
addpath(genpath(pwd)); % --> pwd = '.../figure
params = config_params(); 

for subject = 1:params.numSub
    
    %configure loop settings
    params = config_loop(params,subject);
   
    try
        %load subject data
        [subjectData,params] = load_data(params);

        %loop through interpolation methods (RVGP, SPLINE, REFERENCE,LINEAR)
        FlowFields  = cell(params.numDataTypes,1);
        for d = 1:params.numDataTypes

            %settings  
            cfg = cfg_2dflows(subjectData,params,d);
            
            %compute divergence and curl on EEG vector fields
            cfg.verbose = 0;
            FlowField = compute_2Dflows(cfg);
                   
            % store subject data
            FlowFields{d} = FlowField;

        end
               
    catch
        %record missing subject
        params.missing_subjects(subject) = 1;

        %display error
        disp(['error processing subject',num2str(subject)]);
    end
    
    %save subject level data
    if ~exist(params.paths.output_folder,'dir')
        mkdir(params.paths.output_folder)
    
    end
    cd(params.paths.output_folder)
    save(strcat('s',num2str(params.current_subject),'_FlowFields'),'FlowFields','-v7.3');

        
    % estimate time remaining
    params.meantime(subject) = toc;
    t_remain(mean(params.meantime,'omitnan'),params.numSub-subject);

end

%save parameter settings for script
save('params','params','-v7.3');




