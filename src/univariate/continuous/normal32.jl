"""
    NormalF32(μ,σ)

The *Normal distribution* with mean `μ` and standard deviation `σ≥0` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(x - \\mu)^2}{2 \\sigma^2} \\right)
```

Note that if `σ == 0`, then the distribution is a point mass concentrated at `μ`.
Though not technically a continuous distribution, it is allowed so as to account for cases where `σ` may have underflowed,
and the functions are defined by taking the pointwise limit as ``σ → 0``.

```julia
NormalF32()          # standard Normal distribution with zero mean and unit variance
NormalF32(μ)         # Normal distribution with mean μ and unit variance
NormalF32(μ, σ)      # Normal distribution with mean μ and variance σ^2

params(d)         # Get the parameters, i.e. (μ, σ)
mean(d)           # Get the mean, i.e. μ
std(d)            # Get the standard deviation, i.e. σ
```

External links

* [Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)

"""
struct NormalF32{T <: Real} <: ContinuousFloat32UnivariateDistribution
    μ::T
    σ::T
    NormalF32{T}(μ::T, σ::T) where {T <: Real} = new{T}(μ, σ)
end

#### Outer constructors
NormalF32(μ::Float32, σ::Float32) = NormalF32{Float32}(μ, σ)
NormalF32(μ::T, σ::T) where {T <: Real} = NormalF32{T}(μ, σ)
NormalF32(μ::Real, σ::Real) = NormalF32(Float32(μ), Float32(σ))
NormalF32(μ::Real = 0.0f0) = NormalF32(Float32(μ), one(Float32))

const GaussianF32 = NormalF32

# #### Support
Distributions.minimum(::NormalF32) = -Inf32
Distributions.maximum(::NormalF32) = Inf32

#### Parameters

Distributions.params(d::NormalF32) = (d.μ, d.σ)
@inline Distributions.partype(d::NormalF32{T}) where {T} = T

Distributions.location(d::NormalF32) = d.μ
Distributions.scale(d::NormalF32) = d.σ

Base.eltype(::Type{<:NormalF32}) = Float32

#### Statistics

Distributions.mean(d::NormalF32) = d.μ
Distributions.median(d::NormalF32) = d.μ
Distributions.mode(d::NormalF32) = d.μ

Distributions.var(d::NormalF32) = abs2(d.σ)
Distributions.std(d::NormalF32) = d.σ
Distributions.skewness(d::NormalF32) = zero(Float32)
Distributions.kurtosis(d::NormalF32) = zero(Float32)

function Distributions.entropy(d::NormalF32{T}) where {T}
    return (T(log2π_f32) + one(T)) / T(2) + log(d.σ)
end

#### Evaluation

function Distributions.logpdf(d::NormalF32{T}, x::Real) where {T}
    z = (x - d.μ) / d.σ
    return -(z^2 + T(log2π_f32)) / T(2) - log(d.σ)
end

Distributions.pdf(d::NormalF32, x::Real) = exp(logpdf(d, x))

function Distributions.cdf(d::NormalF32{T}, x::Real) where {T}
    z = (x - d.μ) / (d.σ * T(√2))
    return erfc(-z) / T(2)
end

function Distributions.logcdf(d::NormalF32, x::Real)
    return log(cdf(d, x))
end

Distributions.gradlogpdf(d::NormalF32, x::Real) = (d.μ - x) / d.σ^2

#### Affine transformations

Base.:+(d::NormalF32, c::Real) = NormalF32(d.μ + c, d.σ)
Base.:*(c::Real, d::NormalF32) = NormalF32(c * d.μ, abs(c) * d.σ)

#### Sampling

xval(d::NormalF32, z::Real) = muladd(d.σ, z, d.μ)

Base.rand(rng::AbstractRNG, d::NormalF32) = xval(d, randn(rng, Float32))
function Random.rand!(rng::AbstractRNG, d::NormalF32, A::AbstractArray{Float32})
    randn!(rng, A)
    map!(Base.Fix1(xval, d), A, A)
    return A
end
