@testitem "ContinuousFloat32 Tests" begin
    using Distributions
    # Test the eltype function
    struct TestDistribution <:
        Distributions.Sampleable{Distributions.Univariate, ContinuousFloat32} end
    @test eltype(TestDistribution) == Float32

    # Test the type aliases
    @test ContinuousFloat32UnivariateDistribution ==
        Distributions.Distribution{Distributions.Univariate, ContinuousFloat32}
    @test ContinuousFloat32MultivariateDistribution ==
        Distributions.Distribution{Distributions.Multivariate, ContinuousFloat32}
end

@testitem "DiscreteFloat32 Tests" begin
    using Distributions
    # Test the eltype function
    struct TestDiscreteDistribution <:
        Distributions.Sampleable{Distributions.Univariate, DiscreteFloat32} end
    @test eltype(TestDiscreteDistribution) == Float32

    # Test the type aliases
    @test DiscreteFloat32UnivariateDistribution ==
        Distributions.Distribution{Distributions.Univariate, DiscreteFloat32}
    @test DiscreteFloat32MultivariateDistribution ==
        Distributions.Distribution{Distributions.Multivariate, DiscreteFloat32}
end
