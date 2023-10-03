
% ==============================================================================
%
%        Neural Flows- Reconstruction Comparison , generates figure 3.
%                      Author: Matteo Vinao Carl
%        Affiliation: Imperial College London, Grossman Lab
%                   Date: 29th September 2023
%        Cite: Implicit Gaussian process representation of vector fields 
%        over arbitrary latent manifolds (2023). arXiv:2309.16746

% ==============================================================================
%
%                                 File Requirements:
%       File Name: sx_FlowFields.mat
%       Description: Time resolved divergence and vorticity per subject.
%       
%
%       File Name: boundary_mask.mat
%       Description: mask for convex hull 
%
% ==============================================================================

clc, clear
addpath(genpath(pwd))
params = config_params();


%% get reconstruction errorc
clear reconError
reconError = get_ReconError(params);

%% get exemplar flow field
[SubjectFlowField,Chans] = get_SubjectField(params);

%% vector flow field comparison

clear M
figure(1)
for t = params.video.time2plot
    
    clf
    for d = 1:params.numDataTypes

        XY = params.Coordinates.scaled(Chans,:);

        if strcmp(params.data_types{d},'linear')
            subChans = round(linspace(1,size(XY,1),64));
            XY = XY(subChans,:);
        end
        
    
        %get tangent vectors
        tangent_vectors = squeeze(SubjectFlowField{d}(:,t,:));

        %norm
        normV = vecnorm(tangent_vectors(:,1:2),2,2);
        tangent_vectors = tangent_vectors(:,1:2) ./ normV;


        hold on
    
        %Interpolate the velocity components onto grid
        F_vx = scatteredInterpolant(XY(:,1),...
            XY(:,2),...
            tangent_vectors(:,1), 'linear', 'nearest');
        F_vy = scatteredInterpolant(XY(:,1),...
            XY(:,2), ...
            tangent_vectors(:,2), 'linear', 'nearest');
        
        vx_grid = F_vx(params.Coordinates.Grid.x ,params.Coordinates.Grid.y);
        vy_grid = F_vy(params.Coordinates.Grid.x ,params.Coordinates.Grid.y);

        %get back query points
        vx = griddata(params.Coordinates.Grid.x, params.Coordinates.Grid.y, ...
                    vx_grid, params.Coordinates.scaled(:,1),...
                    params.Coordinates.scaled(:,2));

                %get back query points
        vy = griddata(params.Coordinates.Grid.x, params.Coordinates.Grid.y, ...
                    vy_grid, params.Coordinates.scaled(:,1),...
                    params.Coordinates.scaled(:,2));
    
          
        %get phase
        phases = atan2(vx, vy);

    
        %plot 256 channel vector flow montage
        % Loop over each arrow and plot it with the desired color
        subplot(1,4,params.video.SubplotIndex(d))
        clear yv
        if params.video.color_orientation
    
            hold on
            for i = 1:length(vx)
                 
                idx = dsearchn(params.video.phaseBin,phases(i));
    
                %plot north facing
                yv(i) = quiver(params.Coordinates.scaled(i,1),...
                    params.Coordinates.scaled(i,2),...
                vx(i),vy(i),...
                'k','filled','LineWidth',2, 'MaxHeadSize', 1.5);
                
                yv(i).Color = params.video.colorWheel(idx,:);
                yv(i).MarkerEdgeColor = [0 0 0]; %cfg.colormap(8,:);
                yv(i).MarkerFaceColor = [0 0 0]; %cfg.colormap(8,:);
                yv(i).AutoScaleFactor = 0.2;
        
            end
            %black arrows
            
        else
             yv = quiver(params.Coordinates.scaled(:,1),...
                 params.Coordinates.scaled(:,2),...
            vx,vy,...
            'k','filled','LineWidth',2, 'MaxHeadSize', 1.5);
             yv.AutoScaleFactor = 0.8;
        end
    



        %%
    
        if strcmp(params.data_types{d},'ground')
            title(strcat(params.data_types{d},'truth EEG'));
        else
            title(strcat(params.data_types{d},' reconstruction'));
        end
        grid off
        axis off
        set(gca,'box','off','FontSize',14,...
        'ytick',[],'xtick',[],'xlim',...
        params.video.xlm,'ylim',params.video.ylm);
                
        set(gcf,'color','w','position',...
            [params.video.Xpos,params.video.Ypos,...
            params.video.FSizeX ,params.video.FSizeY]);
        shg


    end


    %%
    h = ['t = ', num2str(t),' ms'];
    text(-15.45, 1.5, h, 'FontSize', 40, 'FontWeight', 'bold'); 
    M(t) = getframe(gcf);
    shg
    drawnow

end


%% Initialize video writer and set properties
%vidObj = VideoWriter('Source_Sink_Comparison.mp4', 'MPEG-4');
vidObj = VideoWriter('Recon_Comparison.mp4', 'MPEG-4');
vidObj.Quality = 100;       % Reduce quality for higher compression
vidObj.FrameRate = 15;     % Set lower frame rate

% Open the video writer object
open(vidObj);

% Loop through frames, write every nth frame and resize each
n = 1;  %  keep every nth frame , 1 = every frame
N = length(M); %number of frames to compute
newHeight = 700;  % New height
newWidth = 3000;   % New width

for i = 1:n:N
    if ~isempty(M(i).cdata)  % Check that the frame is not empty
        % Resize the frame
        resizedFrame = imresize(M(i).cdata, [newHeight, newWidth]);
        
        % Create a new frame structure with resized data
        M_resized = struct('cdata', resizedFrame, 'colormap', M(i).colormap);
        
        % Write the frame to video
        writeVideo(vidObj, M_resized);
    end
end

% Close the video writer object
close(vidObj);



%exportgraphics(gcf,'F-I_color_2rows.png','Resolution',400);

