function SVMClassify = get_Predictors(SVMClassify,params)

    % extract same regions from spline / linear interp/ RVGP reconstructed EEG
    for d = 2:params.numDataTypes
    
        %source and sink predictors
        Seeds = SVMClassify.Reference.Coordinates_ss;
        Data = ClusterRawData.SS{d};
        [Values_ss,Labels_ss] = cs_compileFeatures(Data,Seeds,params);
    
        %spiral predictors
        cfg.seeds = SVMClassify.Reference.Coordinates_ca;
        cfg.data = ClusterRawData.CA{d};
        [Values_ca,Labels_ca] = cs_compileFeatures(cfg);
    
        %compile predictors
        Predictors = [];
        for m = 1:length(Values_ss)
    
           Predictors = cat(2,Predictors,Values_ss{m});
        end
    
        for m = 1:length(Values_ca)
    
           Predictors = cat(2,Predictors,Values_ca{m});
        end
    
    
        SVMClassify.(cfg.data_types{d}).Predictors = Predictors;
        SVMClassify.(cfg.data_types{d}).Coordinates_ss = Coordinates_ss;
        SVMClassify.(cfg.data_types{d}).Labels_ss = Labels_ss;
        SVMClassify.(cfg.data_types{d}).Coordinates_ca = Coordinates_ca;
        SVMClassify.(cfg.data_types{d}).Labels_ca = Labels_ca;
    end
end
