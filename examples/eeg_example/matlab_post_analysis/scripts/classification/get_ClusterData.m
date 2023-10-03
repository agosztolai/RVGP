function ClusterRawData = get_ClusterData(params)
    % Check if file exists
    directory = strcat(params.paths.root_dir,'/mat');
    filename = 'ClusterRawData.mat'; % replace with your actual filename
    filepath = fullfile(directory, filename);
    
    %check if file exists
    if exist(filepath,'file')  ==2
        disp(' loading group level data.');
        load('ClusterRawData.mat').ClusterRawData;
    else %else compute
        disp('ClusterRawData.mat does not exist. Computing from single subject data');

        [CA_ratio,SS_ratio] = config_GroupData(params);
        ClusterRawData.SS = SS_ratio;
        ClusterRawData.CA = CA_ratio;
    
        %save
        save('ClusterRawData','ClusterRawData');
    end

end