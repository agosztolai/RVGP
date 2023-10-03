function [CA_ratio,SS_ratio] = config_GroupData(params)

    % ========================================================================
    %
    % ========================================================================
    %
    % DESCRIPTION:
    % This function loads single subject EEG vector fields and compiles to a
    % single group level structure. 
    %   1. Load Single Subject Data: Import the EEG vector fields per
    %      reconstruction method.
    %   2. Compile Divergence and Curl: Computes mean divergence and 
    %      vorticity per subject
    %   3. Compute Reconstruction Error: Calculates the interpolation error 
    %     relative to the ground truth EEG.
    %
    % INPUT:
    %   - params: output from config_params(). Contains analysis settings.
    %   - sx_FlowField: Single subject flow field. Each cell contains results  
    %     for an interpolation method
    %
    % OUTPUT:
    %   - compiledData: A compiled structure containing reconstruction error, mean
    %                   vorticity, and mean divergence per subject
    %
    % USAGE:
    %   compiledData = config_GroupData(params);
    %
    % WARNING:
    % This function may take a significant amount of time to execute.
    %
    % ==============================================================================
    % Author: [Matteo Vinao Carl]
    % Institution: [Imperial College London]
    % Created: [2 Oct 2023]
    % ==============================================================================
    
    %% WARNING notification to the user
    warning('Loading single subject data. This will take a while...');
    
    % load single subject data
    SS_ratio = cell(params.numDataTypes,1);
    CA_ratio = cell(params.numDataTypes,1);
    for s = 1:params.numSub

        params = config_loop(params,s);
    
        %load subject level flow fields
        clear FlowFields
        load(strcat('s',num2str(s),'_FlowFields'));
        
        %% compute singularities
        for d = 1:params.numDataTypes
        
            % Div (sources and sinks)
            Source = squeeze(mean(FlowFields{d}.Div > ...
                params.clustPerm.ss_threshold,1,'omitnan') ...
                + params.clustPerm.epsilon);
        
            Sink = squeeze(mean(FlowFields{d}.Div ...
                < -params.clustPerm.ss_threshold,1,'omitnan') ...
                + params.clustPerm.epsilon);
                    
            % Curl (clockwise v anti-clockwise)
            Clockwise = squeeze(mean(FlowFields{d}.Cav > ...
                params.clustPerm.ca_threshold,1,'omitnan') ...
                + params.clustPerm.epsilon);
        
            AntiClockwise = squeeze(mean(FlowFields{d}.Cav ...
                < -params.clustPerm.ca_threshold,1,'omitnan') ...
                + params.clustPerm.epsilon);
        
            %compute log ratio
            ss_ratio = log10(squeeze(Source ./ Sink));
            ca_ratio = log10(squeeze(Clockwise ./ AntiClockwise));
        
            %ignore values outside of brain
            ca_ratio(params.boundary_mask == 2) = NaN;
            ss_ratio(params.boundary_mask == 2) = NaN;
        
            %ignore nodes at boundary edge
            ca_ratio(params.boundary_mask == 0) = NaN;
            ss_ratio(params.boundary_mask == 0) = NaN;
        
            %store ratios
            SS_ratio{d}(:,:,s) = ss_ratio;
            CA_ratio{d}(:,:,s) = ca_ratio;
        
        end
    end
end
