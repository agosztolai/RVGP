function FlowField = compute_2Dflows(cfg)

    
    %% COMPUTE FLOW FIELD, DIVERGENCE, AND CURL
    vF.X = cell(cfg.numChunks,1);
    vF.Y =cell(cfg.numChunks,1);
    Div = cell(cfg.numChunks,1);
    Curlz = cell(cfg.numChunks,1);
    Cav = cell(cfg.numChunks,1);
    average_t = [];

    %compute in parallel
    positions3D = parallel.pool.Constant(cfg.positions3D);
    positions2D = parallel.pool.Constant(cfg.positions2D);
    Grid = parallel.pool.Constant(cfg.Grid);

     fprintf('... :: info :: ... \n ..computing 2D tangent vectors \n... divergence and curl...');


    for n = 1:cfg.numChunks
    
        if cfg.verbose
        tic
        end
        thisChunk = cfg.time_chunk(n):cfg.time_chunk(n+1)-1;
    
        % Example usage:
        vectors = parallel.pool.Constant(cfg.VectorField(thisChunk,:,:));
    
        
        clear vFX vFY D Cz Cv tvX tvY tvZ div_scattered curlmag_scattered curlZ_scattered
        parfor t = 1:length(thisChunk)
        
            %compute tanget vectors
            [tangent_vectors] = tanget_vec(positions3D.Value,...
                squeeze(vectors.Value(t,:,:)));
        
            %normalise
            tangent_vectors= tangent_vectors ./ vecnorm(tangent_vectors,2,2);
        
            %Interpolate the velocity components onto this grid
            F_vx = scatteredInterpolant(positions2D.Value(:,1),...
                positions2D.Value(:,2),...
                tangent_vectors(:,1), 'linear', 'none');
            F_vy = scatteredInterpolant(positions2D.Value(:,1),...
                positions2D.Value(:,2), ...
                tangent_vectors(:,2), 'linear', 'none');
        
            vx_grid = F_vx(Grid.Value.x, Grid.Value.y);
            vy_grid = F_vy(Grid.Value.x, Grid.Value.y);
        
            norm_grid = sqrt(vx_grid.^2 + vy_grid.^2);
    
            vx_grid = vx_grid ./ norm_grid;
            vy_grid = vy_grid ./ norm_grid;
        
            vFX(t,:,:) = vx_grid;
            vFY(t,:,:) = vy_grid;


            %% - compute gridded divergence and curl
             % Compute  curl
              D(t,:,:) = divergence(Grid.Value.x,Grid.Value.y,...
                  vx_grid, vy_grid);
                [Cz(t,:,:), Cv(t,:,:)] = curl(Grid.Value.x,Grid.Value.y,...
                    vx_grid, vy_grid);

            %store tanget vectors
            tvX(:,t) = tangent_vectors(:,1);
            tvY(:,t) = tangent_vectors(:,2);
            tvZ(:,t) = tangent_vectors(:,3);

        end
    
    
        tV.X{n} = tvX;
        tV.Y{n} = tvY;
        tV.Z{n} = tvZ;
        vF.X{n} = vFX;
        vF.Y{n} = vFY;
        
        Div{n} = D;
        Curlz{n} = Cz;
        Cav{n} = Cv;

    
        
    
    
        if cfg.verbose
            average_t(n) = toc;
            disp(strcat('estimated time remaining = ',num2str( round((cfg.numChunks - n) * mean(average_t))),'s'));
        end
    
    
    end
    
    
    clear FlowField
        
    FlowField.tvX  = cell2mat(tV.X);
    FlowField.tvY  = cell2mat(tV.Y);
    FlowField.tvZ = cell2mat(tV.Z);

    FlowField.X  = cell2mat(vF.X);
    FlowField.Y  = cell2mat(vF.Y);
    
    FlowField.Div = cell2mat(Div);
    FlowField.CurlZ = cell2mat(Curlz);
    FlowField.Cav = cell2mat(Cav);

end



