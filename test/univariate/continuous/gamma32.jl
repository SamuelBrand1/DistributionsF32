@testitem "GammaF32 construction" begin
    using Distributions

    # Default constructor
    d = GammaF32()
    @test shape(d) === 1.0f0
    @test scale(d) === 1.0f0

    # Single arg
    d = GammaF32(2.0f0)
    @test shape(d) === 2.0f0
    @test scale(d) === 1.0f0

    # Two Float32 args
    d = GammaF32(2.0f0, 3.0f0)
    @test shape(d) === 2.0f0
    @test scale(d) === 3.0f0

    # Integer inputs get converted to Float64 by Distributions.Gamma,
    # but Float32 inputs stay Float32
    d = GammaF32(Float32(2), Float32(3))
    @test partype(d) === Float32

    # Alias check
    @test GammaF32 === Gamma
end

@testitem "GammaF32 no type promotion — params and statistics" begin
    using Distributions

    d = GammaF32(2.0f0, 3.0f0)

    @test params(d) === (2.0f0, 3.0f0)
    @test partype(d) === Float32

    @test mean(d) isa Float32
    @test var(d) isa Float32
    @test std(d) isa Float32
    @test skewness(d) isa Float32
    @test kurtosis(d) isa Float32
    @test mode(d) isa Float32
    @test entropy(d) isa Float32

    @test minimum(d) === 0.0f0
    @test maximum(d) === Inf32

    # Verify known values: Gamma(α=2, θ=3)
    @test mean(d) ≈ 6.0f0        # αθ
    @test var(d) ≈ 18.0f0        # αθ²
    @test mode(d) ≈ 3.0f0        # (α-1)θ
end

@testitem "GammaF32 no type promotion — evaluation" begin
    using Distributions, SpecialFunctions

    d = GammaF32(2.0f0, 1.0f0)

    # logpdf, pdf, cdf should return Float32 when given Float32 input
    @test logpdf(d, 1.0f0) isa Float32
    @test pdf(d, 1.0f0) isa Float32
    @test cdf(d, 1.0f0) isa Float32
    @test logcdf(d, 1.0f0) isa Float32

    # Verify known values for Gamma(2, 1) at x=1:
    # pdf(x) = x * exp(-x)  =>  1 * exp(-1)
    @test pdf(d, 1.0f0) ≈ exp(-1.0f0)
    @test logpdf(d, 1.0f0) ≈ -1.0f0

    # cdf(1) for Gamma(2,1) = 1 - 2*exp(-1) (incomplete gamma)
    @test cdf(d, 1.0f0) ≈ Float32(1 - 2 * exp(-1))

    # Non-standard: Gamma(3, 2) at x=2
    d2 = GammaF32(3.0f0, 2.0f0)
    @test logpdf(d2, 2.0f0) isa Float32
    @test cdf(d2, 2.0f0) isa Float32
end

@testitem "GammaF32 no type promotion — sampling" begin
    using Distributions, Random

    d = GammaF32(2.0f0, 3.0f0)
    rng = Random.default_rng()

    # Single sample is Float32
    s = rand(rng, d)
    @test s isa Float32

    # rand! fills Float32 array
    A = zeros(Float32, 100)
    rand!(rng, d, A)
    @test eltype(A) === Float32
    @test all(isfinite, A)
    @test all(x -> x > 0, A)

    # Test all three sampler branches:
    # shape < 1 (GammaIPSampler)
    d_small = GammaF32(0.5f0, 1.0f0)
    @test rand(rng, d_small) isa Float32

    # shape == 1 (Exponential fallback)
    d_one = GammaF32(1.0f0, 2.0f0)
    @test rand(rng, d_one) isa Float32

    # shape > 1 (GammaMTSampler)
    d_large = GammaF32(5.0f0, 1.0f0)
    @test rand(rng, d_large) isa Float32
end

@testitem "GammaF32 AD: multi-backend gradient of logpdf w.r.t. parameters" setup = [DiffBackends] begin
    # Analytical gradients for Gamma(α=2, θ=1) at x=1.5:
    #   logpdf = (α-1)*log(x) - x/θ - α*log(θ) - loggamma(α)
    #   ∂/∂α = log(x) - log(θ) - digamma(α) = log(1.5) - 0 - digamma(2)
    #   ∂/∂θ = x/θ² - α/θ = 1.5 - 2 = -0.5
    using SpecialFunctions
    α, θ, x = 2.0f0, 1.0f0, 1.5f0
    g_α = Float32(log(x) - log(θ) - digamma(α))
    g_θ = Float32(x / θ^2 - α / θ)

    xv = [α, θ]
    g = [g_α, g_θ]
    contexts = (Constant(GammaF32), Constant(x))
    prep_args = (; x = xv, contexts = contexts)
    scenarios = [
        Scenario{:gradient, :out}(
            logpdf_dist,
            xv,
            contexts...;
            res1 = g,
            prep_args,
            name = "logpdf grad GammaF32 parameters"
        ),
    ]

    test_differentiation(backends, scenarios; correctness = true, detailed = true)
end
