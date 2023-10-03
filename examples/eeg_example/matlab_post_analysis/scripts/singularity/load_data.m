function [subjectData,params] = load_data(params)

        subject_results = load(strcat(params.paths.root_dir,'/flowfields/reconstructed/s',num2str(params.current_subject),'/results.mat'));
        obj_flows = load(strcat(params.paths.root_dir,'/flowfields/raw/s',...
            num2str(params.current_subject),'/obj_flows')).obj_flows;
    
        %get good channels
        params.GChans = {obj_flows.eeg.interp.locs.labels};
    
        %ground truth data
        subjectData.ground.vector_field(:,:,1) = [obj_flows.eeg.ground.data.vx];
        subjectData.ground.vector_field(:,:,2) = [obj_flows.eeg.ground.data.vy];
        subjectData.ground.vector_field(:,:,3) = [obj_flows.eeg.ground.data.vz];
        
        %spline data
        subjectData.spline.vector_field(:,:,1) = [obj_flows.eeg.interp.data.vx];
        subjectData.spline.vector_field(:,:,2) = [obj_flows.eeg.interp.data.vy];
        subjectData.spline.vector_field(:,:,3) = [obj_flows.eeg.interp.data.vz];
        
        %reconstructed data
        subjectData.RVGP.vector_field = subject_results.f_pred.f_pred;

end