function SVMClassify = get_brainSeeds(ClusterRawData,ClusterPermTest,params)

    disp('...')
    disp('Extracting seed regions from reference EEG ...')
    disp('...')
    
    clear SVMClassify
    for d = 1:params.numDataTypes      
        %coordinates for divergence
        params.seed_dataset = 1;
        params.Raw = ClusterRawData.SS{d};
        params.Clusters = ClusterPermTest{d}.Divergence;
        
        [Values_ss,Coordinates_ss,Labels_ss] = cs_brainRegions(params);
           
        %coordiantes for Curl
        params.Raw = ClusterRawData.CA{d};
        params.Clusters = ClusterPermTest{d}.Curl;
        
        [Values_ca,Coordinates_ca,Labels_ca] = cs_brainRegions(params);
        
        %compile predictors
        Predictors = [];
        for m = 1:length(Values_ss)
        
           Predictors = cat(2,Predictors,Values_ss{m});
        end
          
        for m = 1:length(Values_ca)
        
           Predictors = cat(2,Predictors,Values_ca{m});
        end
        
        SVMClassify.(params.data_types{d}).Predictors = Predictors;
        SVMClassify.(params.data_types{d}).Coordinates_ss = Coordinates_ss;
        SVMClassify.(params.data_types{d}).Labels_ss = Labels_ss;
        SVMClassify.(params.data_types{d}).Coordinates_ca = Coordinates_ca;
        SVMClassify.(params.data_types{d}).Labels_ca = Labels_ca;

    end

    %% extract seeds regions from spline / RVGP interpolated data
    for d = 2:params.numDataTypes
    
        %source and sink predictors
        params.seeds = SVMClassify.ground.Coordinates_ss;
        params.data = ClusterRawData.SS{d};

        [Values_ss,Labels_ss] = cs_compileFeatures(params);
    
        %spiral predictors
        params.seeds = SVMClassify.ground.Coordinates_ca;
        params.data = ClusterRawData.CA{d};
        [Values_ca,Labels_ca] = cs_compileFeatures(params);
    
        %compile predictors
        Predictors = [];
        for m = 1:length(Values_ss)
    
           Predictors = cat(2,Predictors,Values_ss{m});
        end
    
        for m = 1:length(Values_ca)
    
           Predictors = cat(2,Predictors,Values_ca{m});
        end
    
    
        SVMClassify.(params.data_types{d}).Predictors = Predictors;
        SVMClassify.(params.data_types{d}).Coordinates_ss = Coordinates_ss;
        SVMClassify.(params.data_types{d}).Labels_ss = Labels_ss;
        SVMClassify.(params.data_types{d}).Coordinates_ca = Coordinates_ca;
        SVMClassify.(params.data_types{d}).Labels_ca = Labels_ca;
    end


end

