module DistributionsF32

using Distributions

include("disttypes.jl")

# export types
export ContinuousFloat32,
    ContinuousFloat32UnivariateDistribution, ContinuousFloat32MultivariateDistribution
export DiscreteFloat32,
    DiscreteFloat32UnivariateDistribution, DiscreteFloat32MultivariateDistribution

end
