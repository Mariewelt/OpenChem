# File with utils for converting xyz-coordingates to z-matix and vice versa
# Adapted from https://github.com/robashaw/geomConvert


import numpy as np


def distance_matrix(xyzarr):
    npart, ncoord = xyzarr.shape
    dist_mat = np.zeros([npart, npart])
    for i in range(npart):
        for j in range(0, i):
            rvec = xyzarr[i] - xyzarr[j]
            dist_mat[i][j] = dist_mat[j][i] = np.sqrt(np.dot(rvec, rvec))
    return dist_mat


def angle(xyzarr, i, j, k):
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi
    return theta


def dihedral(xyzarr, i, j, k, l):
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi


def calculate_zmat(xyzarr):
    distmat = distance_matrix(xyzarr)
    npart, ncoord = xyzarr.shape
    r_list = []
    a_list = []
    d_list = []
    r_connect = []
    a_connect = []
    d_connect = []
    if npart > 0:        
        if npart > 1:
            r_list.append(distmat[0][1])
            r_connect.append(1)
            if npart > 2:
                r_list.append(distmat[0][2])
                a_list.append(angle(xyzarr, 2, 0, 1))
                r_connect.append(1)
                a_connect.append(2)
                if npart > 3:
                    for i in range(3, npart):
                        r_list.append(distmat[i-3][i])
                        a_list.append(angle(xyzarr, i, i-3, i-2))
                        d_list.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                        r_connect.append(i-2)
                        a_connect.append(i-1)
                        d_connect.append(i)
    return r_list, a_list, d_list, r_connect, a_connect, d_connect


def calculate_xyz(radii, angles, dihedrals, r_connect, a_connect, d_connect):
    n_atoms = len(radii) + 1
    xyz_coord = np.zeros([n_atoms, 3])
    if (n_atoms > 1):
        xyz_coord[1] = [radii[0], 0.0, 0.0]
    if (n_atoms > 2):
        i = r_connect[1] - 1
        j = a_connect[0] - 1
        r = radii[1]
        theta = angles[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyz_coord[i]
        b_ij = xyz_coord[j] - xyz_coord[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyz_coord[2] = [x, y, 0.0]

    for n in range(3, n_atoms):
        r = radii[n-1]
        theta = angles[n-2] * np.pi / 180.0
        phi = dihedrals[n-3] * np.pi / 180.0
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = r_connect[n-1] - 1
        j = a_connect[n-2] - 1
        k = d_connect[n-3] - 1
        a = xyz_coord[k]
        b = xyz_coord[j]
        c = xyz_coord[i]
        
        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyz_coord[n] = [new_x, new_y, new_z]
            
    return xyz_coord
