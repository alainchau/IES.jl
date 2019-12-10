# Independent Eigencoordinate Selection

Augmentation of diffusion maps that produces a full-rank embedding.
Based on the following paper. 

Chen, Yu-Chia & Meilă, Marina. (2019). Selecting the independent coordinates of manifolds with large aspect ratios. https://arxiv.org/abs/1907.01651

## Example
```julia
julia> using IES
julia> using Plots; gr()
julia> import Random
julia> Random.seed!(1234);
julia> X = rand(2,1000)
2×1000 Array{Float64,2}:
 0.842877  0.383266  0.415927  0.660104  0.704336  …  0.768678  0.938344  0.458261  0.0464189
 0.512672  0.979221  0.154317  0.560282  0.386316     0.432986  0.466662  0.617414  0.239659
julia> X[1,:] *= 6;
julia> DM = DiffMap(X, m=20, σ=0.1);
Laplacian sparsity = 0.958968
julia> mfd = rmetric(DM, d=2);
julia> S = ies(mfd, :ag, s=2, ζ=0.1, q=true)
2-element Array{Int64,1}:
 1
 8
julia> scatter(DM.Y[1,:], DM.Y[8,:], zcolor=X[1,:], legend=:none)

```
![alt text](https://github.com/alainchau/IES.jl/blob/master/rectangle.png "Logo Title Text 1")



[![Build Status](https://travis-ci.com/alainchau/ies.jl.svg?branch=master)](https://travis-ci.com/alainchau/ies.jl)
[![Codecov](https://codecov.io/gh/alainchau/ies.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/alainchau/ies.jl)
