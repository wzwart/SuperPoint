import cv2 as cv
import numpy as np
from math import cos,sin, pi, sqrt
from shapely.geometry import Polygon , LineString, Point, LinearRing
from shapely.errors import TopologicalError
from shapely.validation import explain_validity


import sys

from math import atan, pi

import matplotlib.pyplot as plt
""" Module used to generate geometrical synthetic shapes """

random_state = np.random.RandomState(None)


def set_random_state(state):
    global random_state
    random_state = state


def get_random_color(background_color):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """
    color = random_state.randint(256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def get_different_color(previous_colors, min_dist=50, max_count=20):
    """ Output a color that contrasts with the previous colors
    Parameters:
      previous_colors: np.array of the previous colors
      min_dist: the difference between the new color and
                the previous colors must be at least min_dist
      max_count: maximal number of iterations
    """
    color = random_state.randint(256)
    count = 0
    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = random_state.randint(256)
    return color


def add_salt_and_pepper(img):
    """ Add salt and pepper noise to an image """
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(noise, 0, 255)
    black = noise < 30
    white = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0
    cv.blur(img, (5, 5), img)
    return np.empty((0, 2), dtype=np.int)


def generate_background(size=(960, 1280), nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=50, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    img = np.zeros(size, dtype=np.uint8)
    dim = max(size)
    cv.randu(img, 0, 255)
    cv.threshold(img, random_state.randint(256), 255, cv.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate([random_state.randint(0, size[1], size=(nb_blobs, 1)),
                            random_state.randint(0, size[0], size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(int(dim * min_rad_ratio),
                                    int(dim * max_rad_ratio)),
                  col, -1)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def generate_custom_background(size, background_color, nb_blobs=3000,
                               kernel_boundaries=(50, 100)):
    """ Generate a customized background to fill the shapes
    Parameters:
      background_color: average color of the background image
      nb_blobs: number of circles to draw
      kernel_boundaries: interval of the possible sizes of the kernel
    """
    img = np.zeros(size, dtype=np.uint8)
    img = img + get_random_color(background_color)
    blobs = np.concatenate([np.random.randint(0, size[1], size=(nb_blobs, 1)),
                            np.random.randint(0, size[0], size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(20), col, -1)
    kernel_size = np.random.randint(kernel_boundaries[0], kernel_boundaries[1])
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def final_blur(img, kernel_size=(5, 5)):
    """ Apply a final Gaussian blur to the image
    Parameters:
      kernel_size: size of the kernel
    """
    cv.GaussianBlur(img, kernel_size, 0, img)


def ccw(A, B, C, dim):
    """ Check if the points are listed in counter-clockwise order """
    if dim == 2:  # only 2 dimensions
        return((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
               > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
    else:  # dim should be equal to 3
        return((C[:, 1, :] - A[:, 1, :])
               * (B[:, 0, :] - A[:, 0, :])
               > (B[:, 1, :] - A[:, 1, :])
               * (C[:, 0, :] - A[:, 0, :]))


def intersect(A, B, C, D, dim):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw(A, C, D, dim) != ccw(B, C, D, dim)) &
                  (ccw(A, B, C, dim) != ccw(A, B, D, dim)))


def ccw_2d(A, B, C):
    """ Check if the points are listed in counter-clockwise order """
    return((C[1] - A[1]) * (B[ 0] - A[0])
           > (B[1] - A[1]) * (C[ 0] - A[0]))

def intersect_2d(A, B, C, D):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw_2d(A, C, D) != ccw_2d(B, C, D)) &
                  (ccw_2d(A, B, C) != ccw_2d(A, B, D)))

def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """
    mask = (points[:, 0] >= 0) & (points[:, 0] < size[1]) &\
           (points[:, 1] >= 0) & (points[:, 1] < size[0])
    return points[mask, :]

def poly_to_np(poly):
    try:
        x,y= poly.exterior.coords.xy
    except:
        print(poly)
        raise

    return np.asarray([np.asarray(x), np.asarray(y)]).T

def polygon_within_polygon(polygon, within_polygon):
    debug_mode=False
    try:
        polygon_as_polygon=Polygon(polygon)
        witin_polygon_as_polygon=Polygon(within_polygon)
        for p in [polygon_as_polygon,witin_polygon_as_polygon]:
            valid=explain_validity(p)
            if not valid=="Valid Geometry":
                print("Invalid Polygon", valid)
                return None

        intersection= Polygon(polygon).intersection(Polygon(within_polygon))
        type= intersection.geom_type

        if str(type) == "MultiPolygon":
            print("deal with this ")
            intersection=intersection[0]

        within_p=np.vstack((within_polygon,within_polygon))
        p=np.vstack((polygon,polygon[0]))
        if debug_mode:
            plt.plot(within_p[:,0],within_p[:,1],c='b')
            plt.plot(p[:,0],p[:,1],c='r')
            plt.title(f"Empty ?{intersection.is_empty:}")
            plt.xlim((0,1280))
            plt.ylim((0,960))
            plt.gca().invert_yaxis()
            plt.show()

    except(TopologicalError):
        within_p=np.vstack((within_polygon,within_polygon))
        p=np.vstack((polygon,polygon[0]))
        if debug_mode:

            plt.plot(within_p[:,0],within_p[:,1],c='b')
            plt.plot(p[:,0],p[:,1],c='r')
            plt.title(f"TopologicalError")
            plt.xlim((0,1280))
            plt.ylim((0,960))
            plt.gca().invert_yaxis()
            plt.show()
    except ValueError as err:
        print(err)
        print("Value error:", sys.exc_info()[0])
        raise()
    except :
        print ("BAD!")
    if intersection.is_empty:
        return None
    else:
        return poly_to_np(intersection)

def line_within_polygon(line, within_polygon):

    # return poly_to_np(LineString(line).intersection(Polygon(within_polygon)))
    return poly_to_np(Polygon(within_polygon).intersection(LineString(line)))



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

def create_bg_polygon(img, max_sides=8):
    """ Draw a polygon with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
    """
    num_corners = random_state.randint(5, max_sides)
    done =False
    while not done:
        x= int(np.mean(random_state.randint(img.shape[1], size=3)))
        y= int(np.mean(random_state.randint(img.shape[0], size=3)))
        size=img.shape[1]+img.shape[0]
        rad = random_state.randint(0.3* size, 0.8*size)
        # Sample num_corners points inside the circle
        angles = sorted(list(random_state.rand(num_corners)*np.pi*2-np.pi))
        points = np.array([[int(x + max(random_state.rand(), 0.2) * rad * cos(a)),
                            int(y + max(random_state.rand(), 0.2) * rad * sin(a))]
                           for a in angles])
        image_bounding_box= np.asarray([[0,0],[0, img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1],0]])
        # print("try")
        try:
            points = polygon_within_polygon(points, image_bounding_box)
            done = not points is None
        except:
            done = False
            pass
    corners = points.reshape((-1, 1, 2)).astype(np.int)
    cv.fillPoly(img, [corners], color=1)
    # for point in points.astype(np.int):
    #     cv.line(img, (x, y), (point[0], point[1]), 12, 3)

    return points


def calc_salient_points_from_line(line, poly_fg, min_angle=15):
    '''
    Calculates the salient points from a line on a foreground polygon

    Args:
        line: The line for which the salient points are determined
        poly_fg: the foreground mask, specified as polygon
        min_angle: minimum angle in degrees between polygon and line to qualify as salient point

    Returns:
        list of points that are salient point of the line on the foreground

    '''
    polygon_segments = []
    for i in range(len(poly_fg) - 1):
        polygon_segments.append([poly_fg[i], poly_fg[i + 1]])
    if not (poly_fg[-1] == poly_fg[-0]).all():
        print("closing the loop")
        polygon_segments.append([poly_fg[-1], poly_fg[0]])
    points = []
    for polygon_segment in polygon_segments:
        if intersect_2d(polygon_segment[0], polygon_segment[1], line[0], line[1]):
            intersection_point, angle = (calculate_intersection(polygon_segment, line))
            if not intersection_point is None:
                if angle > min_angle:
                    points.append(intersection_point)
    if Polygon(poly_fg).contains(Point(line[0])):
        points.append(line[0])
    if Polygon(poly_fg).contains(Point(line[1])):
        points.append(line[1])
    points = np.asarray(points).astype(np.int)
    return points


def calc_salient_points_from_ellipse(x,y,ax,ay,angle, poly_fg, min_angle=22):

    polygon_segments = []
    for i in range(len(poly_fg) - 1):
        polygon_segments.append([poly_fg[i], poly_fg[i + 1]])
    if not (poly_fg[-1] == poly_fg[-0]).all():
        print("closing the loop")
        polygon_segments.append([poly_fg[-1], poly_fg[0]])
    points = []
    for line in polygon_segments:
        a_ps=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
        b_ps= line[0][1]-a_ps*(line[0][0]-x)-y

        rad=angle *pi/180

        A= cos(rad)**2/ax**2 + sin(rad)**2/ay**2
        B= 2* cos(rad)*sin(rad)* (1/ax**2-1/ay**2)
        C= sin(rad)**2/ax**2 + cos(rad)**2/ay**2

        a=A+B*a_ps+C*(a_ps**2)
        b=B*b_ps+2*C*a_ps*b_ps
        c=C*b_ps**2 -1

        D= b**2-4*a*c

        if D > 0 :
            x_0= (-b + sqrt(D))/(2*a)+x
            y_0=a_ps*(x_0-x)+b_ps+y

            xd_0 = (-b + sqrt(D)) / (2 * a)
            yd_0 = a_ps * (xd_0) + b_ps
            rc_0 = (-2 * A * xd_0 - B * yd_0) / (2 * C * yd_0 + B * xd_0)

            L=40

            if within_box([x_0,y_0], line):
                if min_angle<180*abs(atan(rc_0)-atan(a_ps))/pi<180-min_angle:
                    points.append([x_0,y_0])
                    # points.append([x_0+L, y_0+L*rc_0])

            x_1= (-b - sqrt(D))/(2*a)+x
            y_1=a_ps*(x_1-x)+b_ps+y

            xd_1 = (-b - sqrt(D)) / (2 * a)
            yd_1 = a_ps * (xd_1) + b_ps

            rc_1 = (-2 * A * xd_1 - B * yd_1) / (2 * C * yd_1 + B * xd_1)

            if within_box([x_1,y_1], line):
                if min_angle<180*abs(atan(rc_1)-atan(a_ps))/pi<180-min_angle:

                    points.append([x_1,y_1])
                    # points.append([x_1+L, y_1+L*rc_1])
    points = np.asarray(points).astype(np.int)
    return points

def within_range(x, range_in):
    return  (x > range_in[0] and x < range_in[1] and range_in[0] < range_in[1]) or (
            x > range_in[1] and x < range_in[0] and range_in[1] < range_in[0])

def within_box(point, box):
    return  within_range( point[0], [box[0][0],box[1][0]]) and within_range( point[1], [box[0][1],box[1][1]])

def append_points(points, new_points):
    for point_from_line in new_points:
        is_in_new_point = False
        for new_point in points:
            is_in_new_point = is_in_new_point or (new_point == point_from_line).all()
        if not is_in_new_point:
            points = np.concatenate([points, point_from_line.reshape((1, 2))], axis=0)
    return points


def calc_salient_points_from_poly(poly, poly_fg, min_angle=15):
    new_poly = np.empty((0, 2), dtype=np.int)
    # shadow_points = np.empty((0, 2), dtype=np.int)
    for i in range(len(poly)):
        if i == len(poly) - 1:
            line = np.asarray([poly[-1], poly[0]])
        else:
            line = poly[i:i + 2]
        points_from_line = calc_salient_points_from_line(line, poly_fg, min_angle=min_angle)
        new_poly=append_points(new_poly,points_from_line)
        # for point_from_line in points_from_line:
        #     is_in_new_point = False
        #     for new_point in new_poly:
        #         is_in_new_point = is_in_new_point or (new_point == point_from_line).all()
        #     if not is_in_new_point:
        #         new_poly = np.concatenate([new_poly, point_from_line.reshape((1, 2))], axis=0)
        # points_from_line = points_from_line.reshape((-1, 2))
        # shadow_points = np.concatenate([shadow_points, points_from_line.reshape((-1, 2))], axis=0)

    return new_poly


def draw_lines(img, nb_lines=10, fg_poly=None):
    """ Draw random lines and output the positions of the endpoints
    Parameters:
      nb_lines: maximal number of lines
    """
    num_lines = random_state.randint(1, nb_lines)
    segments = np.empty((0, 2,2), dtype=np.int)
    points = np.empty((0, 2), dtype=np.int)
    background_color = int(np.mean(img))
    min_dim = min(img.shape)
    for i in range(num_lines):
        x1 = random_state.randint(img.shape[1])
        y1 = random_state.randint(img.shape[0])
        p1 = np.array([[x1, y1]])
        x2 = random_state.randint(img.shape[1])
        y2 = random_state.randint(img.shape[0])
        p2 = np.array([[x2, y2]])
        # Check that there is no overlap
        if intersect(segments[:, 0,:], segments[:, 1,:], p1, p2, 2):
            continue
        line = np.asarray([[x1,y1],[x2,y2]])
        if not fg_poly is None:
            points_from_line=  calc_salient_points_from_line(line, fg_poly, min_angle=15)
            if not points_from_line is None:
                points_from_line=points_from_line.astype(np.int).reshape((-1,2))
                points= np.concatenate([points, points_from_line ], axis=0)
        else:
            points = np.concatenate([points, line[0],line[1]], axis=0)
        segments = np.concatenate([segments, line.reshape((-1,2,2))], axis=0)
        col = get_random_color(background_color)
        thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
        cv.line(img, (x1, y1), (x2, y2), col, thickness)
    return points



def draw_polygon(img, max_sides=8, fg_poly=None):
    """ Draw a polygon with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
    """
    num_corners = random_state.randint(3, max_sides)
    min_dim = min(img.shape[0], img.shape[1])
    rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
    x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
    y = random_state.randint(rad, img.shape[0] - rad)
    done = False
    counter =0
    while not done:
        print('.')
        counter += 1
        if counter >100:
            return None
        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * pi, num_corners + 1)
        angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                  for i in range(num_corners)]
        points = np.array([[int(x + max(random_state.rand(), 0.4) * rad * cos(a)),
                            int(y + max(random_state.rand(), 0.4) * rad * sin(a))]
                           for a in angles])

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(points[(i-1) % num_corners, :]
                                - points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        points = points[mask, :]
        num_corners = points.shape[0]
        corner_angles = [angle_between_vectors(points[(i-1) % num_corners, :] -
                                               points[i, :],
                                               points[(i+1) % num_corners, :] -
                                               points[i, :])
                         for i in range(num_corners)]
        mask = np.array(corner_angles) < (2 * pi / 3)
        points = points[mask, :]
        num_corners = points.shape[0]
        done = num_corners >=3
        corners = points.reshape((-1, 1, 2))
    if not fg_poly is None:
        points= calc_salient_points_from_poly(points, fg_poly, min_angle=15)
    if len(corners) > 0 :
        cv.polylines(img, [corners], isClosed=True, color=255, thickness=3)
        col = get_random_color(int(np.mean(img)))
        cv.fillPoly(img, [corners,corners[0]],  color=col)

    return points


def overlap(center, rad, centers, rads):
    """ Check that the circle with (center, rad)
    doesn't overlap with the other circles """
    flag = False
    for i in range(len(rads)):
        if np.linalg.norm(center - centers[i]) + min(rad, rads[i]) < max(rad, rads[i]):
            flag = True
            break
    return flag


def angle_between_vectors(v1, v2):
    """ Compute the angle (in rad) between the two vectors v1 and v2. """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def draw_multiple_polygons(img, max_sides=8, nb_polygons=30, fg_poly= None, **extra):
    """ Draw multiple polygons with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
      nb_polygons: maximal number of polygons
    """
    segments = np.empty((0, 4), dtype=np.int)
    centers = []
    rads = []
    points = np.empty((0, 2), dtype=np.int)
    background_color = int(np.mean(img))
    for i in range(nb_polygons):
        num_corners = random_state.randint(3, max_sides)
        min_dim = min(img.shape[0], img.shape[1])
        rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
        x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
        y = random_state.randint(rad, img.shape[0] - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * pi, num_corners + 1)
        angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                  for i in range(num_corners)]
        new_points = [[int(x + max(random_state.rand(), 0.4) * rad * cos(a)),
                       int(y + max(random_state.rand(), 0.4) * rad * sin(a))]
                      for a in angles]
        new_points = np.array(new_points)

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(new_points[(i-1) % num_corners, :]
                                - new_points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        corner_angles = [angle_between_vectors(new_points[(i-1) % num_corners, :] -
                                               new_points[i, :],
                                               new_points[(i+1) % num_corners, :] -
                                               new_points[i, :])
                         for i in range(num_corners)]
        mask = np.array(corner_angles) < (2 * pi / 3)
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        if num_corners < 3:  # not enough corners
            continue


        new_segments = np.zeros((1, 4, num_corners))
        new_segments[:, 0, :] = [new_points[i][0] for i in range(num_corners)]
        new_segments[:, 1, :] = [new_points[i][1] for i in range(num_corners)]
        new_segments[:, 2, :] = [new_points[(i+1) % num_corners][0]
                                 for i in range(num_corners)]
        new_segments[:, 3, :] = [new_points[(i+1) % num_corners][1]
                                 for i in range(num_corners)]

        # Check that the polygon will not overlap with pre-existing shapes
        if intersect(segments[:, 0:2, None],
                     segments[:, 2:4, None],
                     new_segments[:, 0:2, :],
                     new_segments[:, 2:4, :],
                     3) or overlap(np.array([x, y]), rad, centers, rads):
            continue
        centers.append(np.array([x, y]))
        rads.append(rad)
        new_segments = np.reshape(np.swapaxes(new_segments, 0, 2), (-1, 4))
        segments = np.concatenate([segments, new_segments], axis=0)

        # Color the polygon with a custom background
        corners = new_points.reshape((-1, 1, 2))
        mask = np.zeros(img.shape, np.uint8)
        custom_background = generate_custom_background(img.shape, background_color,
                                                       **extra)
        cv.fillPoly(mask, [corners], 255)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = custom_background[locs[0], locs[1]]

        if not fg_poly is None:
            new_points = calc_salient_points_from_poly(new_points, fg_poly, min_angle=15)

        points = np.concatenate([points, new_points], axis=0)
    return points

def line_with_ellipse(a, line):
    ea = LinearRing(a)
    mp = ea.intersection(line)
    if mp.is_empty:
        print('Geometries do not intersect')
        return [], []
    elif mp.geom_type == 'Point':
        return [mp.x], [mp.y]
    elif mp.geom_type == 'MultiPoint':
        return [p.x for p in mp], [p.y for p in mp]
    else:
        raise ValueError('something unexpected: ' + mp.geom_type)

def draw_ellipses(img, nb_ellipses=20, fg_poly= None):
    """ Draw several ellipses
    Parameters:
      nb_ellipses: maximal number of ellipses
    """
    salient_points = np.empty((0, 2), dtype=np.int)
    centers = np.empty((0, 2), dtype=np.int)
    rads = np.empty((0, 1), dtype=np.int)
    min_dim = min(img.shape[0], img.shape[1]) / 4
    background_color = int(np.mean(img))
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = get_random_color(background_color)
        angle = random_state.rand() * 90
        cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360, col, -1)
        if not fg_poly is None:
            new_points= calc_salient_points_from_ellipse(x,y,ax,ay,angle, fg_poly)
            salient_points = append_points(salient_points, new_points)

    return salient_points


def draw_star(img, nb_branches=6, fg_poly=None):
    """ Draw a star and output the interest points
    Parameters:
      nb_branches: number of branches of the star
    """
    num_branches = random_state.randint(3, nb_branches)
    min_dim = min(img.shape[0], img.shape[1])
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
    rad = max(random_state.rand() * min_dim / 2, min_dim / 5)
    x = random_state.randint(rad, img.shape[1] - rad)  # select the center of a circle
    y = random_state.randint(rad, img.shape[0] - rad)
    # Sample num_branches points inside the circle
    slices = np.linspace(0, 2 * pi, num_branches + 1)
    angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
              for i in range(num_branches)]
    points = np.array([[int(x + max(random_state.rand(), 0.3) * rad * cos(a)),
                        int(y + max(random_state.rand(), 0.3) * rad * sin(a))]
                       for a in angles])
    points = np.concatenate(([[x, y]], points), axis=0)

    num_remaining_branches=num_branches
    background_color = int(np.mean(img))
    salient_points = np.empty((0, 2), dtype=np.int)
    for i in range(1, num_remaining_branches + 1):
        col = get_random_color(background_color)
        cv.line(img, (points[0][0], points[0][1]),
                (points[i][0], points[i][1]),
                col, thickness)
        line= np.asarray([points[0],points[i]])
        if not fg_poly is None:
            new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
            salient_points=append_points(salient_points,new_points)
    if not fg_poly is None:
        points= salient_points
    return points


def draw_checkerboard(img, max_rows=7, max_cols=7, fg_poly=None, transform_params=(0.05, 0.15)):
    """ Draw a checkerboard and output the interest points
    Parameters:
      max_rows: maximal number of rows + 1
      max_cols: maximal number of cols + 1
      transform_params: set the range of the parameters of the transformations"""
    background_color = int(np.mean(img))
    # Create the grid
    rows = random_state.randint(3, max_rows)  # number of rows
    cols = random_state.randint(3, max_cols)  # number of cols
    s = min((img.shape[1] - 1) // cols, (img.shape[0] - 1) // rows)  # size of a cell
    x_coord = np.tile(range(cols + 1),
                      rows + 1).reshape(((rows + 1) * (cols + 1), 1))
    y_coord = np.repeat(range(rows + 1),
                        cols + 1).reshape(((rows + 1) * (cols + 1), 1))
    points = s * np.concatenate([x_coord, y_coord], axis=1)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    alpha_affine = np.max(img.shape) * (transform_params[0]
                                        + random_state.rand() * transform_params[1])
    center_square = np.float32(img.shape) // 2
    min_dim = min(img.shape)
    square_size = min_dim // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size,
                       [center_square[0]-square_size, center_square[1]+square_size]])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(-alpha_affine / 2,
                                       alpha_affine / 2,
                                       size=pts1.shape).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(np.concatenate((points,
                                          np.ones(((rows + 1) * (cols + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[0, :2]), axis=1),
                         perspective_transform[0, 2])
    warped_col1 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[1, :2]), axis=1),
                         perspective_transform[1, 2])
    warped_col2 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[2, :2]), axis=1),
                         perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangles
    colors = np.zeros((rows * cols,), np.int32)
    for i in range(rows):
        for j in range(cols):
            # Get a color that contrast with the neighboring cells
            if i == 0 and j == 0:
                col = get_random_color(background_color)
            else:
                neighboring_colors = []
                if i != 0:
                    neighboring_colors.append(colors[(i-1) * cols + j])
                if j != 0:
                    neighboring_colors.append(colors[i * cols + j - 1])
                col = get_different_color(np.array(neighboring_colors))
            colors[i * cols + j] = col
            # Fill the cell
            cv.fillConvexPoly(img, np.array([(warped_points[i * (cols + 1) + j, 0],
                                              warped_points[i * (cols + 1) + j, 1]),
                                             (warped_points[i * (cols + 1) + j + 1, 0],
                                              warped_points[i * (cols + 1) + j + 1, 1]),
                                             (warped_points[(i + 1)
                                                            * (cols + 1) + j + 1, 0],
                                              warped_points[(i + 1)
                                                            * (cols + 1) + j + 1, 1]),
                                             (warped_points[(i + 1)
                                                            * (cols + 1) + j, 0],
                                              warped_points[(i + 1)
                                                            * (cols + 1) + j, 1])]),
                              col)

    # Draw lines on the boundaries of the board at random
    nb_rows = random_state.randint(2, rows + 2)
    nb_cols = random_state.randint(2, cols + 2)
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
    for _ in range(nb_rows):
        row_idx = random_state.randint(rows + 1)
        col_idx1 = random_state.randint(cols + 1)
        col_idx2 = random_state.randint(cols + 1)
        col = get_random_color(background_color)
        line=np.array([warped_points[row_idx * (cols + 1) + col_idx1],
                      warped_points[row_idx * (cols + 1) + col_idx2]])
        cv.line(img, (line[0,0], line[0,1]), (line[1,0], line[1,1]) ,col, thickness)

        cv.line(img, (warped_points[row_idx * (cols + 1) + col_idx1, 0],
                      warped_points[row_idx * (cols + 1) + col_idx1, 1]),
                (warped_points[row_idx * (cols + 1) + col_idx2, 0],
                 warped_points[row_idx * (cols + 1) + col_idx2, 1]),
                col, thickness)


    for _ in range(nb_cols):
        col_idx = random_state.randint(cols + 1)
        row_idx1 = random_state.randint(rows + 1)
        row_idx2 = random_state.randint(rows + 1)
        col = get_random_color(background_color)
        cv.line(img, (warped_points[row_idx1 * (cols + 1) + col_idx, 0],
                      warped_points[row_idx1 * (cols + 1) + col_idx, 1]),
                (warped_points[row_idx2 * (cols + 1) + col_idx, 0],
                 warped_points[row_idx2 * (cols + 1) + col_idx, 1]),
                col, thickness)
    if not fg_poly is None:
        points = np.empty((0, 2), dtype=np.int)
        for row_idx in range(0,rows+1):
            for col_idx in range(0,cols):
                line=np.array([warped_points[row_idx * (cols + 1) + col_idx],
                               warped_points[row_idx * (cols + 1) + col_idx+1]])
                # cv.line(img, (line[0,0], line[0,1]), (line[1,0], line[1,1]) ,col, thickness)
                new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
                points = append_points(points, new_points)

        for row_idx in range(0, rows ):
            for col_idx in range(0, cols+1):
                line=np.array([warped_points[row_idx * (cols + 1) + col_idx],
                               warped_points[(row_idx+1) * (cols + 1) + col_idx]])
                # cv.line(img, (line[0,0], line[0,1]), (line[1,0], line[1,1]) ,col, thickness)
                new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
                points = append_points(points, new_points)
    else:

        # Keep only the points inside the image
        points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_stripes(img, max_nb_cols=13, min_width_ratio=0.04,fg_poly=None,
                 transform_params=(0.05, 0.15)):
    """ Draw stripes in a distorted rectangle and output the interest points
    Parameters:
      max_nb_cols: maximal number of stripes to be drawn
      min_width_ratio: the minimal width of a stripe is
                       min_width_ratio * smallest dimension of the image
      transform_params: set the range of the parameters of the transformations
    """
    background_color = int(np.mean(img))
    # Create the grid
    board_size = (int(img.shape[0] * (1 + random_state.rand())),
                  int(img.shape[1] * (1 + random_state.rand())))
    col = random_state.randint(5, max_nb_cols)  # number of cols
    cols = np.concatenate([board_size[1] * random_state.rand(col - 1),
                           np.array([0, board_size[1] - 1])], axis=0)
    cols = np.unique(cols.astype(int))
    # Remove the indices that are too close
    min_dim = min(img.shape)
    min_width = min_dim * min_width_ratio
    cols = cols[(np.concatenate([cols[1:],
                                 np.array([board_size[1] + min_width])],
                                axis=0) - cols) >= min_width]
    col = cols.shape[0] - 1  # update the number of cols
    cols = np.reshape(cols, (col + 1, 1))
    cols1 = np.concatenate([cols, np.zeros((col + 1, 1), np.int32)], axis=1)
    cols2 = np.concatenate([cols,
                            (board_size[0] - 1) * np.ones((col + 1, 1), np.int32)],
                           axis=1)
    points = np.concatenate([cols1, cols2], axis=0)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    # Prepare the matrices
    alpha_affine = np.max(img.shape) * (transform_params[0]
                                        + random_state.rand() * transform_params[1])
    center_square = np.float32(img.shape) // 2
    square_size = min(img.shape) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size,
                       [center_square[0]-square_size, center_square[1]+square_size]])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(-alpha_affine / 2,
                                       alpha_affine / 2,
                                       size=pts1.shape).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(np.concatenate((points,
                                          np.ones((2 * (col + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[0, :2]), axis=1),
                         perspective_transform[0, 2])
    warped_col1 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[1, :2]), axis=1),
                         perspective_transform[1, 2])
    warped_col2 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[2, :2]), axis=1),
                         perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangles
    color = get_random_color(background_color)
    for i in range(col):
        color = (color + 128 + random_state.randint(-30, 30)) % 256
        cv.fillConvexPoly(img, np.array([(warped_points[i, 0],
                                          warped_points[i, 1]),
                                         (warped_points[i+1, 0],
                                          warped_points[i+1, 1]),
                                         (warped_points[i+col+2, 0],
                                          warped_points[i+col+2, 1]),
                                         (warped_points[i+col+1, 0],
                                          warped_points[i+col+1, 1])]),
                          color)

    # Draw lines on the boundaries of the stripes at random
    nb_rows = random_state.randint(2, 5)
    nb_cols = random_state.randint(2, col + 2)
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
    for _ in range(nb_rows):
        row_idx = random_state.choice([0, col + 1])
        col_idx1 = random_state.randint(col + 1)
        col_idx2 = random_state.randint(col + 1)
        color = get_random_color(background_color)
        cv.line(img, (warped_points[row_idx + col_idx1, 0],
                      warped_points[row_idx + col_idx1, 1]),
                (warped_points[row_idx + col_idx2, 0],
                 warped_points[row_idx + col_idx2, 1]),
                color, thickness)
    for _ in range(nb_cols):
        col_idx = random_state.randint(col + 1)
        color = get_random_color(background_color)
        cv.line(img, (warped_points[col_idx, 0],
                      warped_points[col_idx, 1]),
                (warped_points[col_idx + col + 1, 0],
                 warped_points[col_idx + col + 1, 1]),
                color, thickness)
    if not fg_poly is None:
        points = np.empty((0, 2), dtype=np.int)
        for row_idx in range(0,2):
            for col_idx in range(0,col):
                line=np.array([warped_points[row_idx * (col + 1) + col_idx],
                               warped_points[row_idx * (col + 1) + col_idx+1]])
                cv.line(img, (line[0,0], line[0,1]), (line[1,0], line[1,1]) ,color, thickness)
                new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
                points = append_points(points, new_points)

        for row_idx in range(0, 1 ):
            for col_idx in range(0, col+1):
                line=np.array([warped_points[row_idx * (col + 1) + col_idx],
                               warped_points[(row_idx+1) * (col + 1) + col_idx]])
                cv.line(img, (line[0,0], line[0,1]), (line[1,0], line[1,1]) ,color, thickness)
                new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
                points = append_points(points, new_points)
    else:
        # Keep only the points inside the image
        points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_cube(img, min_size_ratio=0.2, min_angle_rot=pi / 10,
              scale_interval=(0.4, 0.6), trans_interval=(0.5, 0.2),fg_poly=None):
    """ Draw a 2D projection of a cube and output the corners that are visible
    Parameters:
      min_size_ratio: min(img.shape) * min_size_ratio is the smallest achievable
                      cube side size
      min_angle_rot: minimal angle of rotation
      scale_interval: the scale is between scale_interval[0] and
                      scale_interval[0]+scale_interval[1]
      trans_interval: the translation is between img.shape*trans_interval[0] and
                      img.shape*(trans_interval[0] + trans_interval[1])
    """
    # Generate a cube and apply to it an affine transformation
    # The order matters!
    # The indices of two adjacent vertices differ only of one bit (as in Gray codes)
    background_color = int(np.mean(img))
    min_dim = min(img.shape[:2])
    min_side = min_dim * min_size_ratio
    lx = min_side + random_state.rand() * 2 * min_dim / 3  # dimensions of the cube
    ly = min_side + random_state.rand() * 2 * min_dim / 3
    lz = min_side + random_state.rand() * 2 * min_dim / 3
    cube = np.array([[0, 0, 0],
                     [lx, 0, 0],
                     [0, ly, 0],
                     [lx, ly, 0],
                     [0, 0, lz],
                     [lx, 0, lz],
                     [0, ly, lz],
                     [lx, ly, lz]])
    rot_angles = random_state.rand(3) * 3 * pi / 10. + pi / 10.
    rotation_1 = np.array([[cos(rot_angles[0]), -sin(rot_angles[0]), 0],
                           [sin(rot_angles[0]), cos(rot_angles[0]), 0],
                           [0, 0, 1]])
    rotation_2 = np.array([[1, 0, 0],
                           [0, cos(rot_angles[1]), -sin(rot_angles[1])],
                           [0, sin(rot_angles[1]), cos(rot_angles[1])]])
    rotation_3 = np.array([[cos(rot_angles[2]), 0, -sin(rot_angles[2])],
                           [0, 1, 0],
                           [sin(rot_angles[2]), 0, cos(rot_angles[2])]])
    scaling = np.array([[scale_interval[0] +
                         random_state.rand() * scale_interval[1], 0, 0],
                        [0, scale_interval[0] +
                         random_state.rand() * scale_interval[1], 0],
                        [0, 0, scale_interval[0] +
                         random_state.rand() * scale_interval[1]]])
    trans = np.array([img.shape[1] * trans_interval[0] +
                      random_state.randint(-img.shape[1] * trans_interval[1],
                                           img.shape[1] * trans_interval[1]),
                      img.shape[0] * trans_interval[0] +
                      random_state.randint(-img.shape[0] * trans_interval[1],
                                           img.shape[0] * trans_interval[1]),
                      0])
    cube = trans + np.transpose(np.dot(scaling,
                                       np.dot(rotation_1,
                                              np.dot(rotation_2,
                                                     np.dot(rotation_3,
                                                            np.transpose(cube))))))

    # The hidden corner is 0 by construction
    # The front one is 7
    cube = cube[:, :2]  # project on the plane z=0
    cube = cube.astype(int)
    if fg_poly is None:
        points = cube[1:, :]  # get rid of the hidden corner
        points = keep_points_inside(points, img.shape[:2])
    else:
        points = np.empty((0, 2), dtype=np.int)
    # Get the three visible faces
    faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

    # Fill the faces and draw the contours
    col_face = get_random_color(background_color)
    for i in [0, 1, 2]:
        cv.fillPoly(img, [cube[faces[i]].reshape((-1, 1, 2))],
                    col_face)
    thickness = random_state.randint(min_dim * 0.003, min_dim * 0.015)

    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3]:
            col_edge = (col_face + 128
                        + random_state.randint(-64, 64))\
                        % 256  # color that constrats with the face color
            line = np.array([cube[faces[i][j]],
                             cube[faces[i][(j+1)%4]]])
            cv.line(img, (line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]), col_edge, thickness)
            if not fg_poly is None:
                new_points = calc_salient_points_from_line(line, fg_poly, min_angle=15)
                points = append_points(points, new_points)




    return points


def gaussian_noise(img):
    """ Apply random noise to the image """
    cv.randu(img, 0, 255)
    return np.empty((0, 2), dtype=np.int)


def draw_interest_points(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb
