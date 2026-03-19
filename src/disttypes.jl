# Type aliases for Float32 distributions that sit inside the standard
# Distributions.jl hierarchy.  Each concrete distribution overrides
# `Base.eltype` to return Float32 — no custom ValueSupport needed.

const ContinuousFloat32UnivariateDistribution = Distributions.ContinuousUnivariateDistribution
const ContinuousFloat32MultivariateDistribution = Distributions.ContinuousMultivariateDistribution

const DiscreteFloat32UnivariateDistribution = Distributions.DiscreteUnivariateDistribution
const DiscreteFloat32MultivariateDistribution = Distributions.DiscreteMultivariateDistribution
