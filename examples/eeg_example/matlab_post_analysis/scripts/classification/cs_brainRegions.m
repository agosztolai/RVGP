function [Values,Coordinates,Labels] = cs_brainRegions(params)

    %% inputs::
    % (1) Clusters structure output from cs_Clusters
    % -------- Fields:: Clusters, p_values, t_sums, permutation_distribution.


    % (2) Raw structure output from cs_CASS
    % N * N * S matrix (grid X dimension, grid Y dimension, Num subjects)


    Values = []; Labels = [];Coordinates = [];

    dims = size(params.Raw);

    % find all significant clusters
    SigClusters = params.Clusters.Clusters(params.Clusters.p_values < params.clustPerm.alpha);

    %get coordinates from K clusters
    
    numClusters = min(length(SigClusters),params.clustPerm.MaxClusters);
    
    for s = 1:numClusters

        %convert to rows and columns
        [I,J] = ind2sub([dims(1),dims(2)],SigClusters{s});


        %take K samples from clusters
        f = round(linspace(1,length(I),params.clustPerm.numSamples));

        g1 = I(f); g2 = J(f);
        Coordinates{s} = [g1,g2];


        %get histograms
        %features
        AD = []; HC = [];
        for n = 1:length(g1)
            x = squeeze(params.Raw(g1(n),g2(n),params.groups.AD));
            AD = cat(1,AD,x);
            x = squeeze(params.Raw(g1(n),g2(n),params.groups.HC));
            HC = cat(1,HC,x);
        end

        Labels{s} = [ones(length(AD),1);2 * ones(length(HC),1)];

        Values{s} = [AD;HC];
    end


end