

function Coordinates = scale_chanlocs(Chanlocs,GridSize)


    %% -inputs:

    % filename : EEG Chanlocs structures : rsf_Chanlocs.mat
    % Gridsize: 3 element vector with size of grid (x,y,z)

    %%
    GridSizeX = GridSize(1);
    GridSizeY = GridSize(2);
    GridSizeZ = GridSize(3);

    % load chanlocs
    rc = Chanlocs;

    %get labels
    labels = {rc.labels};

    locs = [[rc.X]',[rc.Y]',[rc.Z]'];

    %multi dimentional scaling
    D = pdist(locs,'seuclidean');
    flattened_coordinates = mdscale(D,2);

    %interpolate on 2D sheet
    %interpolate
    Xp = flattened_coordinates(:,1);
    Yp = flattened_coordinates(:,2);
    xvec = linspace(floor(min(Xp(:))), ceil(max(Xp(:))), GridSizeX);
    yvec = linspace(floor(min(Yp(:))), ceil(max(Yp(:))), GridSizeY);
    zvec = linspace(floor(min(Yp(:))), ceil(max(Yp(:))), GridSizeZ);
    
    [X_grid, Y_grid] = meshgrid(xvec, yvec);

        
    [X_grid3,Y_grid3,Z_grid3] = meshgrid(xvec, yvec,zvec);

    Coordinates.scaled = flattened_coordinates;
    Coordinates.locs = locs;
    Coordinates.labels = labels;
    Coordinates.interp.x = xvec;
    Coordinates.interp.y = yvec;
    Coordinates.interp.z = zvec;
    Coordinates.Grid.x = X_grid;
    Coordinates.Grid.y = Y_grid;

    Coordinates.Grid3.x = X_grid3;
    Coordinates.Grid3.y = Y_grid3;
    Coordinates.Grid3.z = Z_grid3;





end