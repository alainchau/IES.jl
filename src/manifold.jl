struct Manifold
    G   # Metric
    H   # Co-metric
    U   # Orthogonal basis      # (d, m, n)
    Σ   # H = U Σ U'
    D   # Diffusion mapping
end

dim(mfd::Manifold) = size(mfd.U, 1)
getm(mfd::Manifold) = size(mfd.U, 2)
getn(mfd::Manifold) = size(mfd.U, 3)

function Base.iterate(mfd::Manifold, state = 1)
    if state > 5
        return nothing
    else
        getfield(mfd, state), state + 1
    end
end

"""
    rmetric(DM::DiffMap, d::Integer, lazy=true)

Estimate the Riemannian metric from data. Set lazy=true if only U is needed.
"""
function rmetric(DM::DiffMap; d::Integer, lazy=true)
    m, n = size(DM)


    # ==== BRUTE FORCE
    # H̃ = zeros(m, m, n)
    # for i in 1:n
    #     for k in 1:m
    #         for ℓ in 1:m
    #             for j in 1:n
    #                 H̃[ℓ, k, i] += DM.L[j, i] * (DM.Y[ℓ, j] - DM.Y[ℓ, i]) * (DM.Y[k, j] - DM.Y[k, i])
    #             end
    #         end
    #     end
    # end

    # ==== Somewhat vectorized but could use improvement
    H̃ = Array{Float64}(undef, m, m, n)

    ydiffcache = Dict{Tuple{Int64, Int64}, Array{Float64}}()
    function ydiff(ℓ, i)
        if !haskey(ydiffcache, (ℓ, i))
            ydiffcache[ℓ, i] = DM.Y[ℓ, :] .- DM.Y[ℓ, i]
        end

        return ydiffcache[ℓ, i]
    end

    for i in 1:n
        for k in 1:m
            for ℓ in k:m
                H̃[ℓ, k, i] = ydiff(ℓ, i)' * (DM.L[:, i] .* ydiff(k, i))
            end
        end
    end
    for k in 1:m
        for ℓ in 1:k-1
            H̃[ℓ, k, :] = H̃[k, ℓ, :]
        end
    end

    U = Array{Float64}(undef, d, m, n)
    Σ = Array{Float64}(undef, d, n)
    H = Array{Float64}(undef, m, m, n)
    G = Array{Float64}(undef, m, m, n)

    for i in 1:n
        # Reduced rank SVD
        tmpU, tmpΣ, _ = svd(H̃[:,:,i])
        U[:,:,i] = permutedims(tmpU[:, 1:d])
        Σ[:,i] = tmpΣ[1:d]

        if !lazy
            H[:,:,i] = U[:,:,i]' * diagm(Σ[:,i]) * U[:,:,i]
            G[:,:,i] = U[:,:,i]' * diagm(Σ[:,i].^-1) * U[:,:,i]
        end
    end

    return Manifold(G, H, U, Σ, DM)
end
