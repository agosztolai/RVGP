function [SubjectFlowField,Chans] = get_SubjectField(params)


    
    %% got flowfields for 4 methods

    clear I G S P
    s = 1; %use subject 1

    config_loop(params,s);

    %load subject level flow fields
    clear FlowFields
    load(strcat('s',num2str(s),'_FlowFields'));
        
    %ground truth data
    G(:,:,1) = FlowFields{1}.tvX;
    G(:,:,2) = FlowFields{1}.tvY;
    G(:,:,3) = FlowFields{1}.tvZ;
    G = G ./ vecnorm(G,2,3);
    
    %spline data
    S(:,:,1) = FlowFields{2}.tvX;
    S(:,:,2) = FlowFields{2}.tvY;
    S(:,:,3) = FlowFields{2}.tvZ;
    S = S ./ vecnorm(S,2,3);
        
    %RVGP 
    P(:,:,1) = FlowFields{3}.tvX;
    P(:,:,2) = FlowFields{3}.tvY;
    P(:,:,3) = FlowFields{3}.tvZ;
    P = P  ./ vecnorm(P,2,3);
    
    I(:,:,1) = FlowFields{4}.tvX;
    I(:,:,2) = FlowFields{4}.tvY;
    I(:,:,3) = FlowFields{4}.tvZ;
    I  = I./ vecnorm(I,2,3);

    % get channel indices
    obj_flows = load(strcat(params.paths.root_dir,'/flowfields/raw/s',...
            num2str(s),'/obj_flows')).obj_flows;

    %get good channels
    Chans = ismember(params.Coordinates.labels,{obj_flows.eeg.interp.locs.labels});
    


    SubjectFlowField = {G,S,P,I};
end