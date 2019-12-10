"""
    bestinc(mfd, k, S0, ζ)

Find the best increment of size k to the set S0 on the manifold mfd with regularization ζ.
"""
function bestinc(mfd::Manifold, k::Integer, S0, ζ)
    best = Dict(:L => -Inf,
                :S => Int[])

    for S in powerset(setdiff(1:getm(mfd), S0), k, k)
        L = iesloss(mfd, S0 ∪ S, ζ)
        if best[:L] < L
            best[:L] = L
            best[:S] = S0 ∪ S
        end
    end
    return best[:S], best[:L]
end


function iesloss(mfd::Manifold, S, ζ)
    n = getn(mfd)
    R = 0.0
    for i in 1:getn(mfd)
        Us = mfd.U[:, S, i]
        R += logabsdet( Us * Us' )[1] / 2n
        R -= sum(log.(mapslices(norm, Us, dims=1) .+ 0.0001)) / n
        # # Quit early if the vectors of Us are collinear
        # R == -Inf && return -Inf
    end
    return R - ζ * sum(mfd.D.Λ[S])
end

iesloss(mfd::Manifold, i::Integer, ζ) = iesloss(mfd, [i], ζ)


function _ies_brute(mfd::Manifold; kwargs...)
    bestinc(mfd, kwargs[:s]-1, [1], kwargs[:ζ])[1]
end

function _ies_sg(mfd::Manifold; kwargs...)
    s    = get(kwargs, :s, 2)
    ζ    = get(kwargs, :ζ, 1.0)
    quick = get(kwargs, :q, true) # need to do this

    d, m, n = size(mfd.U)
    S, L = bestinc(mfd, d-1, [1], ζ)
    𝔏 = [L]
    for _ in d+1:s
        S, L = bestinc(mfd, 1, S, ζ)
        push!(𝔏, L)
        if 𝔏[end-1] > 𝔏[end]
            return S
        end
    end
    return S
end

function _ies_ag(mfd::Manifold; kwargs...)
    # Define sensible defaults
    s    = get(kwargs, :s, 2)
    ζ    = get(kwargs, :ζ, 1.0)
    quick = get(kwargs, :q, true)

    d = dim(mfd)
    m = getm(mfd)

    # Step 1
    # Since the objective is trivial for sets < d, initialize by picking best subset of d indices
    Sk, fSk = bestinc(mfd, d-1, [1], ζ)
    # println(" - > ", Sk, fSk)
    k = 1

    # Step 2
    Δ = zeros(m)
    availableidx = setdiff(1:m, Sk)
    for i in availableidx
        Δ[i] = iesloss(mfd, Sk ∪ i, ζ) - fSk
    end

    selectedthisiteration = zeros(Int8, m)
    while true && k < s-d+1
        # Step c
        imax = 1
        imax2 = 1
        for i in availableidx
            if Δ[i] > Δ[imax]
                imax2 = imax
                imax = i
            elseif Δ[i] > Δ[imax2]
                imax2 = i
            end
        end

        fSk2 = iesloss(mfd, Sk ∪ imax, ζ)
        if selectedthisiteration[imax] == 1
            δ = Δ[imax]
            @goto bottom
        end
        selectedthisiteration[imax] = 1

        # Step d
        δ = fSk2 - fSk
        Δ[imax] = δ
        if δ<Δ[imax2] continue end

        @label bottom
        if quick
            if δ<=0
                println("Returning $Sk")
                return Sk
            end
        end

        Sk = Sk ∪ imax
        setdiff!(availableidx, imax)
        fSk = fSk2
        Δ[imax] = 0
        k += 1
        fill!(selectedthisiteration, 0)
    end

    return Sk
end

const ies_algorithm = Dict(
    :br => _ies_brute,
    :sg => _ies_sg,
    :ag => _ies_ag
)

# """
#     ies
#
# Independent eigencoordinate selection for an embedding of a d-dimensional manifold into R^s.
#
# # Parameters
# -   `X::AbstractArray` Dataset with size Dxn
# -   `k::Integer`: Number of nearest neighbors for each point.
# -   `d::Integer`: Intrinsic dimension for estimating Riemannian metric.
# -   `s::Integer`: Embedding dimension.
# -   `ζ`: Regularizer.
# -   `m`: Dimension of diffusion map embedding. Select s vectors from this set.
# """
# function ies(mfd::Manifold; s=10, ζ=1, which=:ag)
function ies(mfd::Manifold, which=:ag; kwargs...)
    d = dim(mfd)
    @assert d <= kwargs[:s] "Embedding dimension must be at least the intrinsic dimension. $d ≰ $s"
    ies_algorithm[which](mfd; kwargs...)
end
