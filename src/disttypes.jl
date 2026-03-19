struct ContinuousFloat32 <: Distributions.ValueSupport end

function Base.eltype(::Type{<:Distributions.Sampleable{F, ContinuousFloat32}}) where {F}
    return Float32
end

const ContinuousFloat32UnivariateDistribution = Distributions.Distribution{
    Distributions.Univariate, ContinuousFloat32,
}
const ContinuousFloat32MultivariateDistribution = Distributions.Distribution{
    Distributions.Multivariate, ContinuousFloat32,
}

struct DiscreteFloat32 <: Distributions.ValueSupport end

function Base.eltype(::Type{<:Distributions.Sampleable{F, DiscreteFloat32}}) where {F}
    return Float32
end

const DiscreteFloat32UnivariateDistribution = Distributions.Distribution{
    Distributions.Univariate, DiscreteFloat32,
}
const DiscreteFloat32MultivariateDistribution = Distributions.Distribution{
    Distributions.Multivariate, DiscreteFloat32,
}
