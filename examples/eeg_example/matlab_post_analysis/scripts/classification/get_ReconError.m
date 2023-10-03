function reconError = get_ReconError(params)

    % path to file
    directory = strcat(params.paths.root_dir,'/mat');
    filename = 'reconError.mat'; 
    filepath = fullfile(directory, filename);
    
    %check if file exists
    if exist(filepath,'file')  ==2
        disp(' loading group level data.');
        load('reconError.mat');
    else %else compute

        disp('reconError.mat does not exist. Computing ... This may take a while ...');

        ReconError = get_ReconError(params);

        %save
        cd(strcat(params.paths.root_dir,'/mat'))
        save('reconError','reconError');

        load('reconError.mat');
    end

end