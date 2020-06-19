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

import tqdm
import numpy as np

from math import atan, pi

H=960
W=1280
size = (H, W)

dset.set_random_state(np.random.RandomState(0))

bg = dset.generate_background(size)
img_bg=bg.copy()
mid_point= np.asarray([W/2,H/2])
for i in tqdm.tqdm(range(1)):
    with_fg= True
    if with_fg:
        bg = np.zeros(np.array(size))
        fg_poly= dset.create_fg_polygon(bg)
    else:
        bg = np.ones(np.array(size))
        fg_poly = None

    MIN_ANGLE=22.5

    # img=bg.copy()
    # points= dset.draw_polygon(img, fg_poly=fg_poly, min_angle=MIN_ANGLE)
    # img=bg.copy()
    # points= dset.draw_ellipses(img, fg_poly=fg_poly, min_angle=MIN_ANGLE)
    img=bg.copy()
    points= dset.draw_star(img, fg_poly=fg_poly, min_angle=MIN_ANGLE)


# img=bg.copy()
# points= dset.draw_ellipses(img, fg_poly=fg_poly, min_angle=MIN_ANGLE)


dset.draw_checkerboard(img_bg, fg_poly=None, min_angle=MIN_ANGLE)
dset.draw_lines(img_bg, fg_poly=None, min_angle=MIN_ANGLE)


points= dset.finalize_salient_points(points, fg_poly, img, min_angle=MIN_ANGLE)

img=img*bg+(1-bg)*img_bg
# equivalent but more general


n_plots=2
scale = 0.005
fig = plt.figure(tight_layout=True, figsize=(scale*W*n_plots,scale*H))

axs=[fig.add_subplot(1, n_plots, i+1) for i in range(n_plots)]
for ax in axs:
    ax.axis('off')
axs[0].imshow(img, cmap='gray')
n_img=np.zeros((bg.shape[0],bg.shape[1],4), dtype=np.uint8)
n_img[:,:,0]=img
n_img[:,:,1]=img
n_img[:,:,2]=img
n_img[:,:,3]=255

n_bg=np.zeros((bg.shape[0],bg.shape[1],4), dtype=np.uint8)
n_bg[:,:,0]=0
n_bg[:,:,1]=255
n_bg[:,:,2]=0
n_bg[:,:,3]=64*(1-bg)
axs[1].imshow(n_img)
axs[1].imshow(n_bg)

if len(points)>0:
    print (points)
    print (np.min(points[:,0]),np.min(points[:,1]))
    print (np.max(points[:,0]),np.max(points[:,1]))
    print (img.shape)
    print(np.min(points[:,0])>=0, np.min(points[:,1])>=0)
    print(np.max(points[:,0])<=img.shape[1], np.max(points[:,1])<=img.shape[0])
    axs[0].scatter(points[:,0],points[:,1], lw=1, c='r')
    axs[1].scatter(points[:,0],points[:,1], lw=1, c='r')
plt.tight_layout()
plt.show()




