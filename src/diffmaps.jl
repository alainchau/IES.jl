"""

Contains embedding of data under diffusion maps.
"""
struct DiffMap
    Y  # m x n
    L
    Λ
    Φ
end

"""

"""
function DiffMap(X; m::Integer=2, σ = 2)
    D, n = size(X)

    # # Step 1: Compute similarity matrix
    𝒦 = gaussian_similarity(X, σ)

    # Step 2: compute laplacian
    L = rlap(𝒦)

    # Step 3: Compute eigenvectors of L
    # Arpack returns complex-valued eigenvalues.
    Λ, Φ = eigs(L, nev=m+1, which=:SR)
    Λ = real(Λ)
    Φ = real(Φ)

    # Remember that (1,...,1) is an eigenvector with eigenvalue 0. This is uninformative,
    # hence we discard it as usual.
    Φ ./= Φ[:, 1]
    Φ = Φ[:, 2:m+1]
    Λ = Λ[2:m+1]
    Y = permutedims(Φ .* reshape(Λ, (1, :)))
    return DiffMap(Y, L, Λ, Φ)
end

getm(DM::DiffMap) = size(DM.Y, 1)
getn(DM::DiffMap) = size(DM.Y, 2)
size(DM::DiffMap) = size(DM.Y)
size(DM::DiffMap, d::Integer) = size(DM.Y, d)

"""
    Return measure of density for laplacian. (#nonzeros / #total)
"""
sparsity(DM::DiffMap) = sparsity(DM.L)
sparsity(A) = sum(iszero.(A)) / length(A)

function Base.iterate(DM::DiffMap, state = 1)
    if state > 4
        return nothing
    else
        getfield(DM, state), state + 1
    end
end
