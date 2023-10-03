function  params = config_params()


    addpath(genpath(pwd));
    
    disp('#########################################################################')
    disp('#                 ☺☺☺☺ ~~~~ RSVG EEG RECONSTRUCTION  ~~~~ ☺☺☺☺        #')
    disp('#########################################################################')
    disp('------------------------------------------------------------------------')
    fprintf('%s \n', strcat('::Info:: EEG script for interpolation.'))              
    disp('------------------------------------------------------------------------')
    % Tic
    disp('------------------------------------------------------------------------')
    

    % Set default text interpreter to LaTeX
    set(0, 'defaulttextinterpreter', 'latex');
    set(0, 'DefaultTextInterpreter', 'latex');
    set(0, 'DefaultAxesTickLabelInterpreter','latex');
    set(0, 'DefaultLegendInterpreter','latex');

    %paths 
    params.paths.root_dir = pwd ;
    params.paths.extension = 'experiments/rsf_brains/';
    params.paths.subject_folder = '_preproc';
    params.paths.output_folder = strcat(params.paths.root_dir,...
        '/flowfields/div_curl');
    
    %experiemtnal design settings
    params.data_types = {'ground','spline','RVGP','linear'};
    params.behaviour = ...
        load('rsf_participants.mat').rsf_participants;
    params.groups.labels = {'AD','HC'};
    params.groups.AD = ismember(params.behaviour.Diagnosis,'AD');
    params.groups.HC = ismember(params.behaviour.Diagnosis,'HC');
    params.conditions = {'eo','ec'};
    params.subjects = 1:height(params.behaviour);
    
    params.chanlocs = load('rsf_chanlocs.mat').rsf_Chanlocs;
    params.B_mat = load('B_rsf').B_rsf;

   
    %loop sizes
    params.numDataTypes = length(params.data_types);
    params.numSub = length(params.subjects);

     
    %initialise
    params.meantime = nan(params.numSub,1);
    params.missing_subjects = zeros(params.numSub,1);

    %mask
    params.boundary_mask = load('boundary_mask').boundary_mask;


    %cluster permutation testing
    params.clustPerm.ss_threshold = 1.5;
    params.clustPerm.ca_threshold = 0.5;
    params.clustPerm.epsilon = 10^-5;
    params.clustPerm.alpha = 0.05;
    params.clustPerm.nperm = 50000;
    params.clustPerm.MaxClusters = 4; 
    params.clustPerm.numSamples = 20; 
    params.clustPerm.alpha = 0.05;

    %binary classifer
    params.svm.nIter = 50;
    params.svm.nComponents = 3;
    params.svm.verbose= 1;
    params.svm.nIter = 100;

    %plot settings
    params.plots.color_scheme{1} = ...
        [[0.4940 0.1840 0.7560];[0.0350 0.5780 0.840]] ;
    params.plots.color_scheme{2} = ...
        [[0.6350 0.0780 0.1840];[0.0350 0.5780 0.840]] ; 
    params.plots.color_scheme{3} = linspecer(50);
    params.plots.color_scheme{4} = colormap_CD([ 0.45 0.7; 0.08 0.95],[0.86 .45],[0 0],12); %linspecer(50);

        %grid size for interpolation
    params.GridSize = 45 .* ones(1,3);
    params.Coordinates = scale_chanlocs(...
        params.chanlocs,params.GridSize);
    
    %% video settings
    close
    params.video.colorIndex = 1;
    params.video.ClusterMarkerSize = 100;
    params.video.dt = 1000 / 250;
    params.video.time2plot = 1:5000;
    params.video.Xpos = 10;
    params.video.Ypos = 400;
    params.video.FSizeX = 2000;
    params.video.FSizeY = 500;
    params.video.Fz = 44;
    params.video.plot_clusters = 1;
    params.video.scaleFact = 1;  
    params.video.scattercolormap = linspecer(20);
    params.video.colormap = colormap_CD([ 0.45 0.7; 0.08 0.95],[0.86 .45],[0 0],12);
    params.video.ClusterMarkerColor = params.video.colormap(3,:);
    params.video.minClusterSize = 5;
    params.video.xlm = ...
        [floor(min(params.Coordinates.interp.x) * params.video.scaleFact),...
    ceil(max(params.Coordinates.interp.x ) * params.video.scaleFact)];
    params.video.ylm = ...
        [floor(min(params.Coordinates.interp.y) * params.video.scaleFact),...
    ceil(max(params.Coordinates.interp.y) * params.video.scaleFact)];
    params.video.color_orientation = 1;
    params.video.xlm = [-2.02 2.02];
    params.video.ylm = [-2 2];
    params.video.SubplotIndex = [1,4,2,3];
    params.video.N = 256;
    params.video.colorWheel = hsv(256);
    params.video.phaseBin = linspace(-pi,pi,256)';

    close
       

end
