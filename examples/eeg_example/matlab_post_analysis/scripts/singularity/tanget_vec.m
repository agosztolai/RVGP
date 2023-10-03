function [tangent_vectors] = tanget_vec(positions,vectors)
    
    % Convert Cartesian coordinates to spherical coordinates
    r = vecnorm(positions, 2, 2);
    
    % Normalize the vectors
    magnitudes = vecnorm(vectors, 2, 2);
    normalized_vectors = vectors ./ magnitudes;
    
    % Initialize tangent_vectors
    tangent_vectors = zeros(size(vectors));
    
    % Project vectors onto the tangent plane at each point on the sphere
    for i = 1:size(vectors, 1)
        normal_vector = positions(i, :) / r(i);
        projection_matrix = eye(3) - normal_vector' * normal_vector;
        tangent_vectors(i, :) = projection_matrix * normalized_vectors(i, :)';
    end
  
end

