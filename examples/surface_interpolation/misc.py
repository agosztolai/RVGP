#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os


def load_mesh(data='bunny', folder='../data'):
    # Initialize empty lists to store vertices and faces
    vertices = []
    faces = []
    
    file = os.path.join(folder,data)

    # Open the text file in read mode
    with open('{}.obj'.format(file), 'r') as file:

        # Loop through each line in the file
        for line in file:

            # Skip lines starting with '#'
            if line.startswith('#'):
                continue

            # Split the line into words
            words = line.split()

            # Check if it starts with 'v' for vertices or 'f' for faces
            if words[0] == 'v':
                # Extract the numbers and convert them to floats
                vertex = [float(words[1]), float(words[2]), float(words[3])]
                # Append the vertex to the vertices list
                vertices.append(vertex)
            elif words[0] == 'f':
                # Extract the numbers and convert them to integers
                face = [int(words[1]), int(words[2]), int(words[3])]
                # Append the face to the faces list
                faces.append(face)

    # Convert the lists to arrays
    vertices = np.array(vertices)
    faces = np.array(faces)-1
    
    return vertices, faces