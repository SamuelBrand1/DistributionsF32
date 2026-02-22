module DistributionsF32

using Distributions
import Random
using Random: AbstractRNG, randn, randn!
using SpecialFunctions: erfc

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

end
