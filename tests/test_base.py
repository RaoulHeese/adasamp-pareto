# -*- coding: utf-8 -*-
"""Tests."""


import pytest
import numpy as np
from adasamp.sampling import AdaptiveSampler


def _grid_builder(creator_func, N, seed, dim, offset, cut_ref_violation, scale):    
    Y = np.random.RandomState(seed).rand(N,dim) + np.array([offset]*dim)
    Y_ref = np.array([0]*dim)
    Y_pareto = Y[AdaptiveSampler._is_pareto_efficient(Y),:]
    Y_dim = len(Y_ref)
    return AdaptiveSampler._create_pareto_grid(creator_func, Y_pareto, Y_ref, Y_dim, cut_ref_violation, scale), Y_dim
    
def _compare_grids(grid_A, grid_B, Y_dim):
    Y_grid_idx_iter_func_A, Y_grid_map_func_A, Y_grid_dom_func_A, Y_grid_scale_A, Y_grid_size_A = grid_A
    Y_grid_idx_A = [idx for idx in Y_grid_idx_iter_func_A()]
    Y_grid_dom_A = np.array([Y_grid_dom_func_A(idx) for idx in Y_grid_idx_A])
    Y_grid_map_A = np.array([Y_grid_map_func_A(AdaptiveSampler._pareto_grid_map_idx_nodim_to_start_idx(idx, d)) for d in range(Y_dim) for idx in Y_grid_idx_A])
    Y_grid_idx_iter_func_B, Y_grid_map_func_B, Y_grid_dom_func_B, Y_grid_scale_B, Y_grid_size_B = grid_B
    Y_grid_idx_B = [idx for idx in Y_grid_idx_iter_func_B()]
    Y_grid_dom_B = np.array([Y_grid_dom_func_B(idx) for idx in Y_grid_idx_B])
    Y_grid_map_B = np.array([Y_grid_map_func_B(AdaptiveSampler._pareto_grid_map_idx_nodim_to_start_idx(idx, d)) for d in range(Y_dim) for idx in Y_grid_idx_B])
    return Y_grid_idx_A == Y_grid_idx_B, np.all(Y_grid_scale_A == Y_grid_scale_B), Y_grid_size_A == Y_grid_size_B, np.all(Y_grid_dom_A == Y_grid_dom_B), np.all(Y_grid_map_A == Y_grid_map_B)
    
def _build_and_compare_grids(N, seed, dim, offset, cut_ref_violation, scale):
    grid_A, Y_dim = _grid_builder(AdaptiveSampler._create_pareto_grid_cached, N, seed, dim, offset, cut_ref_violation, scale)
    grid_B, _ = _grid_builder(AdaptiveSampler._create_pareto_grid_runtime, N, seed, dim, offset, cut_ref_violation, scale)
    return _compare_grids(grid_A, grid_B, Y_dim)
    
def test_grid_creation_2d():
    assert _build_and_compare_grids(N=10, seed=42, dim=2, offset=1, cut_ref_violation=False, scale=True)
    
def test_grid_creation_3d():
    assert _build_and_compare_grids(N=10, seed=42, dim=3, offset=1, cut_ref_violation=False, scale=True) 
    
def test_grid_creation_4d():
    assert _build_and_compare_grids(N=10, seed=42, dim=4, offset=1, cut_ref_violation=False, scale=True)
    