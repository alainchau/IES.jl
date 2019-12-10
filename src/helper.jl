"""
    rlap(A)

Compute renormalized graph Laplacian for adjacency matrix A.
"""
function rlap(A)
    W = diagm(A * ones(size(A, 1)))
    L̃ = W \ (W \ A)'
    W̃ = diagm(L̃ * ones(size(L̃, 1)))

    println("Laplacian sparsity = $(sparsity(W̃ \ (W̃ - L̃)))")
    return sparse(W̃ \ (W̃ - L̃))
end


"""
    gaussian_similarity(X, σ=1)

Compute similarity using Gaussian kernel. Enforces sparsity by setting the similarity of
distant points to zero.
"""
function gaussian_similarity(X, σ=1)
    𝒦 = pairwise(Euclidean(), X, dims=2)
    n = size(X, 2)
    for i in 1:n
        for j in i+1:n
            𝒦[j, i] = 𝒦[j, i] <= 3σ ? exp(-(𝒦[j,i]/σ)^2) : 0
            𝒦[i, j] = 𝒦[j, i]
        end
    end
    𝒦[diagind(𝒦)] .= 1

    s = sparsity(𝒦)
    if s < 0.8
        println("Warning. sparsity = $s")
    end

    return 𝒦
end
