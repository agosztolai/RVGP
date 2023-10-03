function get_PermData(ClusterRawData,params)
    % Check if file exists
    directory = strcat(params.paths.root_dir,'/mat');
    filename = 'ClusterPermTest.mat'; % replace with your actual filename
    filepath = fullfile(directory, filename);
    
    %check if file exists
    if exist(filepath,'file')  ==2
        disp(' loading group level data.');
        load('ClusterPermTest.mat').ClusterPermTest;
    else %else compute

        disp('ClusterPermTest.mat does not exist. Computing ... This may take a while ...');

        ClusterPermTest= cell(params.numDataTypes,1);

        for d = 1:params.numDataTypes

            params.interp_method = d;

            %cluster permutation test per node ( vorticity and divergence)
            [Divergence,Curl] = cs_Clusters(ClusterRawData,params);
        
            ClusterPermTest{d}.Divergence = Divergence;
            ClusterPermTest{d}.Curl = Curl;
            
        end

        %save
        cd(strcat(params.paths.root_dir,'/mat'))
        save('ClusterPermTest','ClusterPermTest');

        load('ClusterPermTest.mat');
    end

end