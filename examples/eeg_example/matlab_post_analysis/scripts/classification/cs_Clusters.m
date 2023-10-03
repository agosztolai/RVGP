function [Divergence,Curl] = cs_Clusters(ClusterRawData,params)

    %Patients
    AD = ismember(params.behaviour.Diagnosis,'AD');

    %healthy controls
    HC = ismember(params.behaviour.Diagnosis,'HC');

    %sources
    G1  = ClusterRawData.SS{params.interp_method}(:,:,AD);
    G2  = ClusterRawData.SS{params.interp_method}(:,:,HC);

    [clusters, p_values, t_sums, permutation_distribution ] = ...
        permutest_fullperm( G1, G2, 0, ...
        params.clustPerm.alpha, params.clustPerm.nperm, 1);


        
    %store restults
    Divergence.Clusters = clusters;
    Divergence.p_values = p_values;
    Divergence.t_sums = t_sums;
    Divergence.permutation_distribution = ...
        permutation_distribution;


    %sinks
        
    G1  = ClusterRawData.CA{params.interp_method}(:,:,AD);
    G2  = ClusterRawData.CA{params.interp_method}(:,:,HC);

    %store restulsts
    [clusters, p_values, t_sums, permutation_distribution ] = ...
        permutest_fullperm( G1, G2,0, ...
        params.clustPerm.alpha, params.clustPerm.nperm, 1);


    Curl.Clusters = clusters;
    Curl.p_values = p_values;
    Curl.t_sums = t_sums;
    Curl.permutation_distribution = ...
        permutation_distribution;


end