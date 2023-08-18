import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def regular_square_mesh_to_obj(verticies, wrap_x=False, wrap_y=False):
    # assumes verticies are regularly spaced on a 2d grid and that you want th eUV map to be square.
    n, m = verticies.shape[:2]
    n_uv = n + 1 if wrap_x else n
    m_uv = m + 1 if wrap_y else m

    ii, jj = np.meshgrid(np.arange(n_uv), np.arange(m_uv))
    ii = ii.T / (n_uv - 1)
    jj = jj.T / (m_uv - 1)

    uv_coords = np.stack([ii, jj], axis=-1)
    uv_coords = np.flip(uv_coords, axis=-1)

    return square_mesh_to_obj(verticies, uv_coords, wrap_x=wrap_x, wrap_y=wrap_y)


def square_mesh_to_obj(verticies, uv_coords=None, wrap_x=False, wrap_y=False):
    """Turns a mesh based on quad faces as a pushforward from a 2D square to some mesh.
    Assumes faces are built from the oredering of the verticies.

    Parameters
    ----------
    verticies : np.ndarry
        (n, m, 3) array of vertex coords
    uv_coords : np.ndarry, optional
        ([n/n+1] if wrap_x, [m/m+1] if wrap_y), 2, UV coords. Assumes you've added the
        extra uv coord manually.
    wrap_x : bool, optional
        if the x edge gets wrapped around
    wrap_y : bool, optional
        if the y edge gets wrapped around
    """

    n, m = verticies.shape[:2]

    # build faces from verticies
    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    face_index = ii + n * jj

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, j, 0] = face_index[i, j]
            faces[i, j, 1] = face_index[(i + 1) % n, j]
            faces[i, j, 2] = face_index[(i + 1) % n, (j + 1) % m]
            faces[i, j, 3] = face_index[i, (j + 1) % m]

    n_uv = n + wrap_x
    m_uv = m + wrap_y
    ii, jj = np.meshgrid(np.arange(n_uv), np.arange(m_uv))
    ii = ii.T
    jj = jj.T
    uv_index = ii + n_uv * jj
    uv_indicies = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            uv_indicies[i, j, 0] = uv_index[i, j]
            uv_indicies[i, j, 1] = uv_index[i + 1, j]
            uv_indicies[i, j, 2] = uv_index[i + 1, j + 1]
            uv_indicies[i, j, 3] = uv_index[i, j + 1]

    lines = []

    lines.append("# Verticies")

    for j in range(m):
        for i in range(n):
            lines.append(
                f"v {verticies[i, j, 0]:0.5f} {verticies[i, j, 1]:0.5f} {verticies[i, j, 2]:0.5f}"
            )

    if uv_coords is not None:
        lines.append("")
        lines.append("# UV Coords")

        for j in range(m_uv):
            for i in range(n_uv):
                lines.append(f"vt {uv_coords[i, j, 0]:0.5f} {uv_coords[i, j, 1]:0.5f}")

    lines.append("# Faces")
    if uv_coords is not None:
        for j in range(m_faces):
            for i in range(n_faces):
                lines.append(
                    f"f {faces[i, j, 0]+1}/{uv_indicies[i, j, 0]+1} {faces[i, j, 1]+1}/{uv_indicies[i, j, 1]+1} {faces[i, j, 2]+1}/{uv_indicies[i, j, 2]+1} {faces[i, j, 3]+1}/{uv_indicies[i, j, 3]+1}"
                )
    else:
        for j in range(m_faces):
            for i in range(n_faces):
                lines.append(
                    f"f {faces[i,j,0]+1} {faces[i,j,1]+1} {faces[i,j,2]+1} {faces[i,j,3]+1}"
                )

    return lines


def mesh_to_obj(verticies, faces, uv_coords=None):
    lines = []

    lines.append("# Verticies")

    for i in range(verticies.shape[0]):
        lines.append(
            f"v {verticies[i, 0]:0.5f} {verticies[i, 1]:0.5f} {verticies[i, 2]:0.5f}"
        )

    if uv_coords is not None:
        lines.append("")
        lines.append("# UV Coords")

        for i in range(uv_coords.shape[0]):
            lines.append(f"vt {uv_coords[i, 0]:0.5f} {uv_coords[i, 1]:0.5f}")

    lines.append("")

    lines.append("# Faces")
    if uv_coords is not None:
        if faces.shape[1] == 3:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1}/{faces[i,0]+1} {faces[i,1]+1}/{faces[i,1]+1} {faces[i,2]+1}/{faces[i,2]+1}"
                )
        elif faces.shape[1] == 4:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1}/{faces[i,0]+1} {faces[i,1]+1}/{faces[i,1]+1} {faces[i,2]+1}/{faces[i,2]+1} {faces[i,3]+1}/{faces[i,3]+1}"
                )
    else:
        if faces.shape[1] == 3:
            for i in range(faces.shape[0]):
                lines.append(f"f {faces[i,0]+1} {faces[i,1]+1} {faces[i,2]+1}")
        elif faces.shape[1] == 4:
            for i in range(faces.shape[0]):
                lines.append(
                    f"f {faces[i,0]+1} {faces[i,1]+1} {faces[i,2]+1} {faces[i,3]+1}"
                )

    return lines


def make_faces_from_vectors(verticies, vectors):

    vert_lines = []

    face_lines = []

    for i in range(vectors.shape[0]):
        vert = verticies[i]
        vec_norm = jnp.linalg.norm(vectors[i])
        if vectors[i, 2] != 0.0:
            orth1 = jnp.array([0, 1, -vectors[i, 1] / vectors[i, 2]])
            orth1 = orth1 / jnp.linalg.norm(orth1)
            orth2 = jnp.cross(vectors[i], orth1)
            orth2 = orth2 / jnp.linalg.norm(orth2)
        else:
            orth1 = jnp.array([0, 1, 0])
            orth2 = jnp.array([1, 0, 0])

        vert_pp = vert + vec_norm * (orth1 + orth2)
        vert_lines.append(f"v {vert_pp[0]:0.5f} {vert_pp[1]:0.5f} {vert_pp[2]:0.5f}")
        vert_pm = vert + vec_norm * (orth1 - orth2)
        vert_lines.append(f"v {vert_pm[0]:0.5f} {vert_pm[1]:0.5f} {vert_pm[2]:0.5f}")
        vert_mm = vert + vec_norm * (-orth1 - orth2)
        vert_lines.append(f"v {vert_mm[0]:0.5f} {vert_mm[1]:0.5f} {vert_mm[2]:0.5f}")
        vert_mp = vert + vec_norm * (-orth1 + orth2)
        vert_lines.append(f"v {vert_mp[0]:0.5f} {vert_mp[1]:0.5f} {vert_mp[2]:0.5f}")

        face_lines.append(f"f {i*4 + 1} {i*4 + 2} {i*4 + 3} {i*4 + 4}")

    lines = []

    lines.append("# Verticies")

    lines += vert_lines

    lines += ["", "# Faces"]

    lines += face_lines

    return lines


def make_verticies_from_vectors(verticies, vectors):

    vert_lines = []

    face_lines = []

    for i in range(vectors.shape[0]):
        vert = verticies[i]
        vec_norm = jnp.sqrt(jnp.linalg.norm(vectors[i]))
        if vectors[i, 2] != 0.0:
            orth1 = jnp.array([0, 1, -vectors[i, 1] / vectors[i, 2]])
            orth1 = orth1 / jnp.linalg.norm(orth1)
            orth2 = jnp.cross(vectors[i], orth1)
            orth2 = orth2 / jnp.linalg.norm(orth2)
        else:
            orth1 = jnp.array([0, 1, 0])
            orth2 = jnp.array([1, 0, 0])

        vert_pp = vert + vec_norm * (orth1 + orth2)
        vert_lines.append(f"v {vert_pp[0]:0.5f} {vert_pp[1]:0.5f} {vert_pp[2]:0.5f}")
        vert_pm = vert + vec_norm * (orth1 - orth2)
        vert_lines.append(f"v {vert_pm[0]:0.5f} {vert_pm[1]:0.5f} {vert_pm[2]:0.5f}")
        vert_mm = vert + vec_norm * (-orth1 - orth2)
        vert_lines.append(f"v {vert_mm[0]:0.5f} {vert_mm[1]:0.5f} {vert_mm[2]:0.5f}")
        vert_mp = vert + vec_norm * (-orth1 + orth2)
        vert_lines.append(f"v {vert_mp[0]:0.5f} {vert_mp[1]:0.5f} {vert_mp[2]:0.5f}")

        face_lines.append(f"f {i*4 + 1} {i*4 + 2} {i*4 + 3} {i*4 + 4}")

    lines = []

    lines.append("# Verticies")

    lines += vert_lines

    lines += ["", "# Faces"]

    lines += face_lines

    return lines


def make_scalar_texture(function, locations, file, cmap="viridis"):
    locations_shape = locations.shape
    locations = locations.reshape((-1, *locations_shape[2:]))
    values = function(locations)
    values = values.reshape(locations_shape[:2])
    plt.imsave(file, values, cmap=cmap)


def save_obj(lines, file):
    with open(file, "w") as f:
        for line in lines:
            f.write(line + "\n")


def export_vec_field(V, vecs, file):
    np.savetxt(file, jnp.concatenate([V, vecs], axis=1), delimiter=",")
