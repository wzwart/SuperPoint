import sys
from os.path import dirname
sys.path.append(dirname(r"C:\Data Science\Thesis\ba_keypoints\Superpoint_TF\\"))
import matplotlib.pyplot as plt

from superpoint.datasets.utils import augmentation_legacy as daug
from superpoint.datasets import synthetic_dataset_bg as dset
from notebooks.utils import plot_imgs
from shapely.geometry import Polygon, LineString, Point

import cv2 as cv
import time

import numpy as np

from math import atan, pi

H=960
W=1280
size = (H, W)



n_show = 2

img = dset.generate_background(size)


mid_point= np.asarray([W/2,H/2])

points = 12

angles = points * 2


def calculate_intersection(line_a,line_b):

    line_1= np.asarray(line_a)
    line_2= np.asarray(line_b)
    # if any of the two lines are vertical:
    if (line_1[0,0]==line_1[1,0]) or (line_2[0,0]==line_2[1,0]):
        # swap x and y
        flipped = True
    else:
        flipped = False
    if flipped:
        line_1=np.flip(line_1, axis=1)
        line_2=np.flip(line_2, axis=1)
    a_1=(line_1[0,1]-line_1[1,1])/(line_1[0,0]-line_1[1,0])
    b_1= line_1[0,1]-a_1*line_1[0,0]
    a_2=(line_2[0,1]-line_2[1,1])/(line_2[0,0]-line_2[1,0])
    b_2= line_2[0,1]-a_2*line_2[0,0]
    if a_2==a_1:
        print("Lines are parallel, returning None")
        return None,0
    angle = np.abs(180*(atan(a_1)-atan(a_2))/pi)
    x=(b_1-b_2)/(a_2-a_1)
    y=a_1*x+b_1
    if flipped:
        return (y,x), angle
    else:
        return (x,y), angle

def ccw(A, B, C):
    """ Check if the points are listed in counter-clockwise order """
    return((C[1] - A[1]) * (B[ 0] - A[0])
           > (B[1] - A[1]) * (C[ 0] - A[0]))

def intersect(A, B, C, D):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw(A, C, D) != ccw(B, C, D)) &
                  (ccw(A, B, C) != ccw(A, B, D)))

def calc_salient_points(line, poly_fg, min_angle):
    '''
    Calculates the salient points from a line on a foreground polygon

    Args:
        line: The line for which the salient points are determined
        poly_fg: the foreground mask, specified as polygon
        min_angle: minimum angle in degrees between polygon and line to qualify as salient point

    Returns:
        list of points that are salient point of the line on the foreground

    '''
    polygon_segments=[]
    for i in range(len(poly_fg) - 1):
        polygon_segments.append([poly_fg[i], poly_fg[i + 1]])
    if not (poly_fg[-1] == poly_fg[-0]).all():
        print("closing the loop")
        polygon_segments.append([poly_fg[-1], poly_fg[0]])
    points=[]
    for polygon_segment in polygon_segments:
        if intersect(polygon_segment[0],polygon_segment[1],line[0],line[1]):
            intersection_point,angle = (calculate_intersection(polygon_segment, line))
            if not intersection_point is None:
                if angle > min_angle:
                    points.append(intersection_point)
    if Polygon(poly_fg).contains(Point(line[0])):
        points.append(line[0])
    if Polygon(poly_fg).contains(Point(line[1])):
        points.append(line[1])
    points=np.asarray(points)
    np.random.shuffle(points)
    # sort the points
    if (line[0,0]==line[1,0]): # if we are dealing with a vertical line
        #sort over the y axis
        points = np.flip(np.sort(np.flip(points,axis=1),axis=0),axis=1)
    else:
    #     # sort over the x axis
        points.sort(axis=0)
    return points


def create_star(center, num_points, inner_radius, outer_radius):
    center_x = center[0]
    center_y = center[1]
    points=[]
    for i in range(num_points):
        alpha= i * 2* pi/(num_points)
        points.append(list(
            [center_x + outer_radius * np.cos(alpha),
             center_y + outer_radius * np.sin(alpha)]
        ))
        alpha= (i+0.5) * 2* pi/(num_points)
        points.append(list(
            [center_x + inner_radius * np.cos(alpha),
             center_y + inner_radius * np.sin(alpha)]
        ))
    points.append(points[0])
    return np.asarray(points)

# star= create_star(mid_point,num_points=12,inner_radius=100, outer_radius=600)
#
# line = np.asarray([[0,H/4],[W,1.2*H/4]])
# line = np.asarray([[W/4,0],[W/4,H]])
# line = np.asarray([[W/2,H/2],[3*W/4,H]])
#
# points = calc_salient_points(line=line, poly_fg=star, min_angle=3)

#
# plt.plot(star[:,0],star[:,1])
# plt.plot(line[:,0],line[:,1])
# if len(points)>0:
#     plt.scatter(points[:,0],points[:,1],c='g')
# plt.gca().invert_yaxis()
# plt.xlim((0,W))
# plt.ylim((0,H))
# plt.show()
#
# print (points)
#




bg= np.zeros(np.array(size))

bg_pts= dset.create_bg_polygon(bg)

print(Polygon(bg_pts))

# points= dset.draw_lines(img, fg_poly=bg_pts)
points= dset.draw_ellipses(img,fg_poly=bg_pts)

ax = plot_imgs([img, bg*img], ylabel=["Background"],  normalize=True, cmap='gray')
if len(points)>0:
    print (points)
    print (np.min(points[:,0]),np.min(points[:,1]))
    print (np.max(points[:,0]),np.max(points[:,1]))
    print (img.shape)
    print(np.min(points[:,0])>=0, np.min(points[:,1])>=0)
    print(np.max(points[:,0])<=img.shape[1], np.max(points[:,1])<=img.shape[0])
    ax[0].scatter(points[:,0],points[:,1], lw=1, c='r')
    ax[1].scatter(points[:,0],points[:,1], lw=1, c='r')
    ax[0].plot([0,0,1280,1280],[0,960,960,0], lw=4)
    ax[1].plot([0,0,1280,1280],[0,960,960,0], lw=4)

plt.show()



from sympy import Point, Ellipse
from sympy import Ellipse, Point, Line, sqrt, Segment
from sympy.plotting import plot

e = Ellipse(Point(0, 0), 5, 7)

p_1=e.intersection(Line(Point(0,0), Point(0, 1)))
p_3=e.intersection(Segment(Point(6,0), Point(6, 1)))
p_2=e.intersection(Segment(Point(5,0), Point(5.1, 1)))

lines= e.tangent_lines(p_2[0])
print(lines)
eq= lines[0].equation()
print(eq)
print(p_1)
print(p_2)

