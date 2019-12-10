module IES

import Base: iterate, size
using LinearAlgebra
using Distances: pairwise, Euclidean
using Combinatorics: powerset
using Arpack: eigs
using SparseArrays

include("diffmaps.jl")
include("helper.jl")
include("manifold.jl")
include("ies.jl")

export rlap, DiffMap, gaussian_similarity, rmetric, ies

end
