import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def _plot_grid(xv, yv, squares, ax):
    for i  in np.linspace(0, xv.shape[1] - 1, squares+1, dtype=int):
        ax.plot(xv[i,:], yv[i,:], 'k-')
    for j in np.linspace(0, xv.shape[0] - 1, squares+1, dtype=int):
        ax.plot(xv[:,j], yv[:,j], 'k-')
    
    ax.axis('off')

def _radial_distortion(xv, yv, k):
    xv_radial = np.zeros_like(xv)
    yv_radial = np.zeros_like(yv)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            r = np.sqrt(xv[i,j]**2 + yv[i,j]**2)
            radial = (1 + (k[0]*(r**2) + k[1]*(r**4) + k[2]*(r**6)))/(1 + (k[3]*(r**2) + k[4]*(r**4) + k[5]*(r**6)))
            xv_radial[i,j] = xv[i,j]*radial
            yv_radial[i,j] = yv[i,j]*radial
    return xv_radial, yv_radial

def _tangetial_distortion(xv, yv, p):
    xv_tang = np.zeros_like(xv)
    yv_tang = np.zeros_like(yv)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            x = xv[i,j]
            y = yv[i,j]
            r = np.sqrt(x**2 + y**2)
            x_tang = x + (2*p[0]*x*y + p[1]*(r**2 + 2*x**2))
            y_tang = y + (p[0]*(r**2 + 2*y**2) + 2*p[1]*x*y)
            xv_tang[i,j] = x_tang
            yv_tang[i,j] = y_tang
    return xv_tang, yv_tang

def plot_distortion(k1:float,k2:float,k3:float,k4:float,k5:float,k6:float, p1:float,p2:float) -> None:
    """Plots radial, tangential and compounded (radial + tangential) distortion grid. Using the Brown-Conrady model.

    Args:
        k1 (float): radial distortion coefficient
        k2 (float): radial distortion coefficient
        k3 (float): radial distortion coefficient
        k4 (float): radial distortion coefficient
        k5 (float): radial distortion coefficient
        k6 (float): radial distortion coefficient
        p1 (float): tangential distortion coefficient
        p2 (float): tangential distortion coefficient
    """
    k = (k1,k2,k3,k4,k5,k6)
    p = (p1,p2)
    squares = 10 # amount of squares in the grid
    pts = 100
    # realistical values for image with 2500 x 2500 pixels with focal length of 35mm which is close to 10500 pixels with basler camera pixel size, origin is in the center - therfore x and y should be within values +-(2500/10500)/2
    width = 0.23
    height = 0.23
    xv, yv = np.meshgrid(np.linspace(-width/2,width/2,pts), np.linspace(-height/2,height/2,pts))

    xv_radial, yv_radial = _radial_distortion(xv, yv, k)
    xv_tang, yv_tang = _tangetial_distortion(xv, yv, p)

    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    _plot_grid(xv_radial, yv_radial, squares, axs[0])
    axs[0].set_title('Radial distortion grid')

    _plot_grid(xv_tang, yv_tang, squares, axs[1])
    axs[1].set_title('Tangential distortion grid')

    _plot_grid(xv_radial + xv_tang, yv_radial + yv_tang, squares, axs[2])
    axs[2].set_title('Compounded distortion grid')
    plt.show()