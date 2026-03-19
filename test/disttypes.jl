@testitem "ContinuousFloat32 type aliases" begin
    using Distributions
    # The type aliases should map to the standard Distributions.jl types
    @test ContinuousFloat32UnivariateDistribution === Distributions.ContinuousUnivariateDistribution
    @test ContinuousFloat32MultivariateDistribution === Distributions.ContinuousMultivariateDistribution

    # NormalF32 should be a ContinuousUnivariateDistribution
    @test NormalF32 <: Distributions.ContinuousUnivariateDistribution
    @test eltype(NormalF32) === Float32
end

@testitem "DiscreteFloat32 type aliases" begin
    using Distributions
    # The type aliases should map to the standard Distributions.jl types
    @test DiscreteFloat32UnivariateDistribution === Distributions.DiscreteUnivariateDistribution
    @test DiscreteFloat32MultivariateDistribution === Distributions.DiscreteMultivariateDistribution

    # PoissonF32 should be a DiscreteUnivariateDistribution
    @test PoissonF32 <: Distributions.DiscreteUnivariateDistribution
    @test eltype(PoissonF32) === Float32
end
