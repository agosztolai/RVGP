function cfg = cfg_2dflows(subjectData,params,d)


    disp('------------------------------------------------------------------------');
    fprintf('%s \n', strcat('EEG interpolation method :: ',  params.data_types{d}));       
    disp('------------------------------------------------------------------------');
    
    
    %% settings for computing diveregence / curl
    clear cfg

    %compute on ground truth, spline recon or RVGP recon
    if ~strcmp(params.data_types{d},'linear')

    %multi dimentional scaled coordinates 
    cfg.SC = params.Coordinates.scaled;

    %X and Y scaled coordinates
    cfg.positions2D = params.Coordinates.scaled(...
        ismember(params.Coordinates.labels,params.GChans),:);

    %original 3D positions
    cfg.positions3D = params.Coordinates.locs(...
        ismember(params.Coordinates.labels,params.GChans),:);

    %interpolated grid points
    cfg.interp = params.Coordinates.interp;
    cfg.Grid = params.Coordinates.Grid;
    cfg.VectorField = subjectData.(params.data_types{d}).vector_field;
    
    %do in chunks to avoid preallocation
    cfg.resamples = 1:10000; %use 40 seconds of EEG data (sr = 250Hz)
    cfg.time_chunk = 1:100:max(cfg.resamples)+1;
    cfg.numChunks = length(cfg.time_chunk)-1;
    
    %resample
    cfg.VectorField = cfg.VectorField(cfg.resamples,:,:);
    cfg.verbose = 0;

    %compute on linear interp
    elseif strcmp(params.data_types{d},'linear')

        % % interpolate vector field, compute divergence and curl
        clear cfg
        cfg.subChanIndex = round(linspace(1,length(params.GChans),64));
        cfg.subChanLabels = params.GChans(cfg.subChanIndex);
        cfg.Coordinates = params.Coordinates;
        cfg.positions2D = params.Coordinates.scaled(...
            ismember(params.Coordinates.labels,cfg.subChanLabels),:);
        cfg.positions3D = params.Coordinates.locs(...
            ismember(params.Coordinates.labels,cfg.subChanLabels),:);
        cfg.interp = params.Coordinates.interp;
        cfg.Grid = params.Coordinates.Grid;
        cfg.gchans = params.GChans;
        cfg.subChanIndex = cfg.subChanIndex;
        cfg.subChanLabels = cfg.subChanLabels;

        %do in chunks to avoid preallocation
        cfg.resamples = 1:10000;
        cfg.time_chunk = 1:100:max(cfg.resamples)+1;
        cfg.numChunks = length(cfg.time_chunk)-1;
        
        %resample
        cfg.VectorField = ...
            subjectData.ground.vector_field(cfg.resamples,cfg.subChanIndex,:);
    end

end