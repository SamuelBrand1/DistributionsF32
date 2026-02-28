"""
    PoissonF32(λ)

The *Poisson distribution* with rate parameter `λ` has probability mass function

```math
P(X = k) = \\frac{\\lambda^k}{k!} e^{-\\lambda}, \\quad \\text{for } k = 0, 1, 2, \\ldots.
```

All parameters and samples are represented in `Float32` arithmetic.
Sampling uses the Ahrens-Dieter (1982) algorithm adapted from
[PoissonRandom.jl](https://github.com/SciML/PoissonRandom.jl).

```julia
PoissonF32()         # Poisson distribution with rate parameter 1
PoissonF32(lambda)   # Poisson distribution with rate parameter lambda

params(d)            # Get the parameters, i.e. (λ,)
mean(d)              # Get the mean arrival rate, i.e. λ
```

External links

* [Poisson distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_distribution)
"""
struct PoissonF32{T <: Real} <: DiscreteFloat32UnivariateDistribution
    λ::T
    PoissonF32{T}(λ::Real) where {T <: Real} = new{T}(λ)
end

PoissonF32(λ::T) where {T <: Real} = PoissonF32{T}(λ)
PoissonF32() = PoissonF32{Float32}(1.0f0)

#### Support
Distributions.minimum(::PoissonF32) = zero(Float32)
Distributions.maximum(::PoissonF32) = Inf32

Base.eltype(::Type{<:PoissonF32}) = Float32

### Parameters

Distributions.params(d::PoissonF32) = (d.λ,)
@inline Distributions.partype(::PoissonF32{T}) where {T} = T

rate(d::PoissonF32) = d.λ

### Statistics

Distributions.mean(d::PoissonF32) = d.λ
Distributions.mode(d::PoissonF32) = floor(Float32, d.λ)
Distributions.var(d::PoissonF32) = d.λ
Distributions.skewness(d::PoissonF32) = one(typeof(d.λ)) / sqrt(d.λ)
Distributions.kurtosis(d::PoissonF32) = one(typeof(d.λ)) / d.λ

### Evaluation

function Distributions.logpdf(d::PoissonF32{T}, x) where {T <: Real}
    return xlogy(x, d.λ) - d.λ - loggamma(x + one(T))
end


# CDF via the regularized upper incomplete gamma function:
#   P(X ≤ k) = Q(k+1, λ) where (P,Q) = gamma_inc(a, x)
function Distributions.cdf(d::PoissonF32, x::Real)
    k = floor(Float32, x)
    _, Q = gamma_inc(max(k, zero(k)) + one(k), d.λ)
    return ifelse(k < 0, zero(d.λ), Q)
end

function Distributions.ccdf(d::PoissonF32, x::Real)
    k = floor(Float32, x)
    P, _ = gamma_inc(max(k, zero(k)) + one(k), d.λ)
    return ifelse(k < 0, one(d.λ), P)
end

### Sampling
#
# Adapted from PoissonRandom.jl (SciML) for Float32 arithmetic.
# Algorithm: J.H. Ahrens, U. Dieter (1982)
# "Computer Generation of Poisson Deviates from Modified Normal Distributions"
# ACM Transactions on Mathematical Software, 8(2):163-179

const _INV_SQRT_2PI_F32 = Float32(inv(sqrt(2π)))

# Simple counting method for small λ (O(λ))
function _pois_count_rand(rng::AbstractRNG, λ::Float32)
    n = 0
    c = randexp(rng, Float32)
    while c < λ
        n += 1
        c += randexp(rng, Float32)
    end
    return Float32(n)
end

# Procedure F (Float32 version)
function _pois_procf(λ::Float32, K::Float32, s::Float32)
    ω = _INV_SQRT_2PI_F32 / s
    b1 = inv(24.0f0 * λ)
    b2 = 0.3f0 * b1 * b1
    c3 = inv(7.0f0) * b1 * b2
    c2 = b2 - 15.0f0 * c3
    c1 = b1 - 6.0f0 * b2 + 45.0f0 * c3
    c0 = 1.0f0 - b1 + 3.0f0 * b2 - 15.0f0 * c3

    if K < 10
        px = -λ
        py = λ^K / Float32(factorial(Int(K)))
    else
        δ = inv(12.0f0 * K)
        δ -= 4.8f0 * δ^3
        V = (λ - K) / K
        px = K * log1pmx(V) - δ
        py = _INV_SQRT_2PI_F32 / sqrt(K)
    end
    X = (K - λ + 0.5f0) / s
    X2 = X^2
    fx = X2 / -2.0f0
    fy = ω * (((c3 * X2 + c2) * X2 + c1) * X2 + c0)
    return px, py, fx, fy
end

# Ahrens-Dieter method for large λ (≥ 6)
function _pois_ad_rand(rng::AbstractRNG, λ::Float32)
    s = sqrt(λ)
    d = 6.0f0 * λ^2
    L = floor(Float32, λ - 1.1484f0)
    # Step N
    G = λ + s * randn(rng, Float32)

    if G >= 0.0f0
        K = floor(Float32, G)
        # Step I
        if K >= L
            return K
        end

        # Step S
        U = rand(rng, Float32)
        if d * U >= (λ - K)^3
            return Float32(K)
        end

        # Step P
        px, py, fx, fy = _pois_procf(λ, K, s)

        # Step Q
        if fy * (1.0f0 - U) <= py * exp(px - fx)
            return Float32(K)
        end
    end

    while true
        # Step E
        E = randexp(rng, Float32)
        U = 2.0f0 * rand(rng, Float32) - 1.0f0
        T = 1.8f0 + copysign(E, U)
        if T <= -0.6744f0
            continue
        end

        K = floor(Float32, λ + s * T)
        px, py, fx, fy = _pois_procf(λ, K, s)
        c = 0.1069f0 / λ

        # Step H
        @fastmath if c * abs(U) <= py * exp(px + E) - fy * exp(fx + E)
            return Float32(K)
        end
    end
    return
end

function Base.rand(rng::AbstractRNG, d::PoissonF32)
    λ = Float32(d.λ)
    return λ < 6.0f0 ? _pois_count_rand(rng, λ) : _pois_ad_rand(rng, λ)
end
