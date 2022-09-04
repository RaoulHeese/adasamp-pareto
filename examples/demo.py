# -*- coding: utf-8 -*-
"""Helper classes for adaptive sampling demos."""


import numpy as np
import matplotlib.pyplot as plt


def circle_simulator_func(X, power=2):
    circle_centers = [[1,0], [0,1]]
    circle_radii = [.5,.5]
    num_points = X.shape[0]
    Y = []
    f = []
    for i in range(num_points):
        x = np.ravel(X[i,:])
        hits = [np.linalg.norm(c - x) <= r for (c, r) in zip(circle_centers, circle_radii)]
        feasibility = np.any(hits)
        opt_goal = np.array(x)
        if hits[0]:
            opt_goal += [0,.5]
        if hits[1]:
            opt_goal += [.5,0]
        opt_goal = np.power(opt_goal, power)
        Y.append(opt_goal)
        f.append(feasibility)
    return (np.array(Y).reshape(num_points,2), np.array(f).reshape(num_points))


def circle_pareto_func(N):
    return np.concatenate((np.concatenate((np.linspace(0, .5, N).reshape(N,1),np.linspace(1, 1.5, N).reshape(N,1))),np.concatenate((np.array(np.sqrt(.25-(np.linspace(0, .5, N).reshape(N,1)-0)**2)+1).reshape(N,1),np.array(np.sqrt(.25-(np.linspace(1, 1.5, N).reshape(N,1)-1)**2)+0).reshape(N,1)))), axis=1)
    

def circle_callback_func(sampler, X, Y, f, iteration):
    # options
    grid_resolution = 20
    figsize = (10, 10)
    show_true_pareto = True
    show_path = True
    workers = 1
    
    # plot
    plt.figure(figsize=figsize)
    if grid_resolution > 0:
        x_grid = np.linspace(sampler._X_limits[0][0], sampler._X_limits[0][1], grid_resolution)
        y_grid = np.linspace(sampler._X_limits[1][0], sampler._X_limits[1][1], grid_resolution).T
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        Z = np.c_[x_grid.ravel(), y_grid.ravel()]
        def parallel_opt(Z):
            opt = []
            for z in Z:
                if sampler._opt_func is not None:
                    opt.append(sampler._opt_func(z.ravel(), workers=workers))
                else:
                    opt.append(0)
            return np.array(opt)
        utility = parallel_opt(Z).reshape(x_grid.shape)
        plt.contourf(x_grid, y_grid, utility, cmap="coolwarm", alpha=.25)     
    if show_true_pareto:
        X_true_pareto = circle_pareto_func(N=512)
        plt.scatter(X_true_pareto[:,0], X_true_pareto[:,1], c='c', s=15) 
    plt.scatter(X[f==0,0], X[f==0,1], c='r')
    plt.scatter(X[f==1,0], X[f==1,1], c='g')  
    X_pareto = X[f==1,:][sampler._is_pareto_efficient(Y[f==1,:]),:]
    plt.scatter(X_pareto[:,0], X_pareto[:,1], c='b')
    plt.scatter(X[:sampler._initial_samples,0], X[:sampler._initial_samples,1], facecolor='none', edgecolor='k', s=110)
    if iteration is not None:
        plt.scatter(X[-sampler._virtual_iterations:,0], X[-sampler._virtual_iterations:,1], facecolor='none', edgecolor='c', marker='s', s=110)
    if show_path:
        plt.plot(X[sampler._initial_samples:,0], X[sampler._initial_samples:,1], c='b', alpha=.05)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    title = "[initial sampling]" if iteration is None else "[iteration {:d}]".format(iteration)
    plt.title(title)
    plt.show()
    
    # return test callback result
    return iteration