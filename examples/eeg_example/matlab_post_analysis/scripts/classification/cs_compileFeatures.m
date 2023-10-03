function [Values,Labels] = cs_compileFeatures(params)


    numClusters = length(params.seeds);

    Labels = cell(1,numClusters);
    Values = cell(1,numClusters);
    
    for s = 1:numClusters
        AD = []; HC = [];
        for n = 1:size(params.seeds{s},1)
    
            %get SS predictors
            x = squeeze(params.data(params.seeds{s}(n,1),...
                params.seeds{s}(n,2),params.groups.AD));
    
            AD  = cat(1,AD,x);

            %get CA predictors
            x = squeeze(params.data(params.seeds{s}(n,1),...
                params.seeds{s}(n,2),params.groups.HC));

            HC  = cat(1,HC,x);
        end

        Labels{s} = [ones(length(AD),1);2 * ones(length(HC),1)];

        Values{s} = [AD;HC];

    end





end
