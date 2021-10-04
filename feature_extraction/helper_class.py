#!/usr/bin/python
import open3d as o3d
import numpy as np
import random
from sklearn.cluster import DBSCAN


def NumpyToPCD(xyz):
    """Function to convert numpy array to open3d point cloud 
        Input: xyz - numpy array
        Output: pcd - Point Cloud data"""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def detect_planes(input_pcd, min_ratio):
    """Function to detect multiple planes using RANSAC algorithm"""
    pcd = np.asarray(input_pcd.points)

    inplanes = []
    dist_threshold=0.005
    ransac_n=20
    num_iter=1000

    N = len(pcd)
    count = 0

    while (count < (1 - min_ratio)*N):
        pcd = NumpyToPCD(pcd)
        plane_model, inliers = pcd.segment_plane(dist_threshold, ransac_n, num_iter)
        count += len(inliers)
        pcd = np.asarray(pcd.points)
        inplanes.append((plane_model, pcd[inliers]))
        pcd = np.delete(pcd, inliers, axis=0)
    
    return inplanes

def feature_calc(result):
    # Initialize required arrays
    planes = []
#     colors = []
    equations = []
    tilt_angle = []

    # Target plane to calculate plane angle
    xz_plane = np.array([0, 1, 0])

    for eq, plane in results:

        # Initiate random rgb values for different planes
#         r = random.random()
#         g = random.random()
#         b = random.random()

#         color = np.zeros((plane.shape[0], plane.shape[1]))
#         color[:, 0] = r
#         color[:, 1] = g
#         color[:, 2] = b

        # Calculating tilt angles of the planes detected
        vec1 = [eq[0], eq[1], eq[2]]
        num = np.dot(vec1,xz_plane)
        denom = np.linalg.norm(vec1)*np.linalg.norm(xz_plane)
        theta = np.arccos(num/denom)

        # Adding the values to the appropriate arrays
        tilt_angle.append(theta)
        planes.append(plane)
#         colors.append(color)
        equations.append(eq)

    # Reshaping the array for clustering algorithm
    tilt_angle = np.asarray(tilt_angle).reshape(-1,1)

    # Using clustering algorithm to identify anomalies
    cluster = DBSCAN(eps=0.03,min_samples=1).fit(tilt_angle)
    labels = cluster.labels_

    # Returning the labels with most counts and identifying its index
    label,counts = np.unique(labels,return_counts = True)
    index = np.where(counts==np.max(counts))[0][0]
    cluster_index = label[index]

    # Initializing required arrays
    eq_result = []
    points = []
#     rgb = []
    count = 0

    # Retrieving the equation of the planes that correspond to labels from clustering algorithm
    for i in labels:
        if (i == cluster_index):
            eq_result.append(equations[count])
            points.append(planes[count])
#             rgb.append(colors[count])
        count += 1

    # Concatenating the arrays for visualization purposes
    points = np.concatenate(points, axis=0)
#     rgb = np.concatenate(rgb, axis=0)

    # Visualization of the Point Cloud data
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)
    # o3d.visualization.draw_geometries([pcd])

    # Calculating the Tread and Riser dimensions
    y_mean = np.average(points[:,1])
    z_mean = np.average(points[:,2])
    y25 = np.percentile(points[:,1],25)
    y75 = np.percentile(points[:,1],75)
    z25 = np.percentile(points[:,2],25)
    z75 = np.percentile(points[:,2],75)

    y1 = []
    y2 = []
    y3 = []
    z1 = []
    z2 = []
    z3 = []

    for a,b,c,d in eq_result:
        y1.append((-c*z_mean - d)/b) 
        z1.append((-y_mean*b - d)/c)
        y2.append((-c*z25 - d)/b) 
        z2.append((-y25*b - d)/c)
        y3.append((-c*z75 - d)/b) 
        z3.append((-y75*b - d)/c)
    #     print("y = {}x + {}".format(-b/c,-d/c))


    tread = ((np.max(z1) - np.min(z1)) + (np.max(z2) - np.min(z2)) + (np.max(z3) - np.min(z3))) * (1/3)
    riser = ((np.max(y1) - np.min(y1)) + (np.max(y2) - np.min(y2)) + (np.max(y3) - np.min(y3))) * (1/3)
    
    return tread,riser