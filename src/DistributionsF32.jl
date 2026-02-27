module DistributionsF32

using Distributions
using LogExpFunctions: xlogy, log1pmx
import Random
using Random: AbstractRNG, randn, randn!, randexp
using SpecialFunctions: erfc, loggamma, gamma_inc

include("disttypes.jl")

# export types
export ContinuousFloat32,
    ContinuousFloat32UnivariateDistribution, ContinuousFloat32MultivariateDistribution
export DiscreteFloat32,
    DiscreteFloat32UnivariateDistribution, DiscreteFloat32MultivariateDistribution

# Float32 constants
const log2π_f32 = Float32(log(2π))

# Univariate continuous distributions
include("univariate/continuous/normal32.jl")
export NormalF32, GaussianF32

# Univariate discrete distributions
include("univariate/discrete/poisson32.jl")
export PoissonF32

end
