@testitem "PoissonF32 construction" begin
    using Distributions

    # Default constructor
    d = PoissonF32()
    @test d.λ === 1.0f0

    # Float32 input
    d = PoissonF32(5.0f0)
    @test d.λ === 5.0f0

    # Float64 input preserved (enables AD with Dual numbers)
    d = PoissonF32(5.0)
    @test d.λ === 5.0
end

@testitem "PoissonF32 no type promotion — params and statistics" begin
    using Distributions

    d = PoissonF32(5.0f0)

    @test params(d) === (5.0f0,)
    @test partype(d) === Float32
    @test eltype(PoissonF32) === Float32

    @test mean(d) isa Float32
    @test mode(d) isa Float32
    @test var(d) isa Float32
    @test skewness(d) isa Float32
    @test kurtosis(d) isa Float32

    @test mean(d) === 5.0f0
    @test mode(d) === 5.0f0
    @test var(d) === 5.0f0
    @test skewness(d) ≈ 1.0f0 / sqrt(5.0f0)
    @test kurtosis(d) ≈ 0.2f0

    @test minimum(d) === 0.0f0
    @test maximum(d) === Inf32
end

@testitem "PoissonF32 no type promotion — evaluation" begin
    using Distributions

    d = PoissonF32(5.0f0)

    # Type checks
    @test logpdf(d, 3.0f0) isa Float32
    @test pdf(d, 3.0f0) isa Float32
    @test cdf(d, 3.0f0) isa Float32
    @test ccdf(d, 3.0f0) isa Float32
    @test logcdf(d, 3.0f0) isa Float32
    @test logccdf(d, 3.0f0) isa Float32

    # Known values (compared against Distributions.Poisson reference)
    @test logpdf(d, 3.0f0) ≈ -1.9634458f0
    @test pdf(d, 3.0f0) ≈ 0.1403739f0

    # CDF / CCDF
    @test cdf(d, 0.0f0) ≈ 0.006737947f0
    @test cdf(d, 5.0f0) ≈ 0.61596066f0
    @test ccdf(d, 0.0f0) ≈ 0.99326205f0
    @test ccdf(d, 5.0f0) ≈ 0.38403934f0

    # Boundary: x < 0
    @test cdf(d, -1.0f0) === 0.0f0
    @test ccdf(d, -1.0f0) === 1.0f0

    # CDF + CCDF = 1
    @test cdf(d, 3.0f0) + ccdf(d, 3.0f0) ≈ 1.0f0
end

@testitem "PoissonF32 no type promotion — sampling" begin
    using Distributions, Random

    # Small λ (count_rand path)
    d_small = PoissonF32(2.0f0)
    rng = Random.MersenneTwister(42)
    s = rand(rng, d_small)
    @test s isa Float32
    @test s == floor(s)  # integer-valued

    # Large λ (ad_rand path)
    d_large = PoissonF32(20.0f0)
    s = rand(rng, d_large)
    @test s isa Float32
    @test s == floor(s)

    # Sample mean convergence
    rng2 = Random.MersenneTwister(123)
    d = PoissonF32(5.0f0)
    samples = [rand(rng2, d) for _ in 1:100_000]
    @test eltype(samples) === Float32
    @test all(x -> x >= 0, samples)
    @test abs(sum(samples) / length(samples) - 5.0f0) < 0.1f0
end

@testitem "PoissonF32 AD: multi-backend gradient of logpdf w.r.t. rate" setup =
    [DiffBackends] begin
    # For Poisson(λ), logpdf(k, λ) = k*log(λ) - λ - loggamma(k+1)
    # ∂/∂λ logpdf = k/λ - 1
    # At λ=5, k=3: gradient = 3/5 - 1 = -0.4

    xv = [5.0f0]
    g = [-0.4f0]
    contexts = (Constant(PoissonF32), Constant(3.0f0))
    prep_args = (; x = xv, contexts = contexts)
    scenarios = [
        Scenario{:gradient, :out}(
            logpdf_dist,
            xv,
            contexts...;
            res1 = g,
            prep_args,
            name = "logpdf grad PoissonF32 rate",
        ),
    ]

    test_differentiation(backends, scenarios; correctness = true, detailed = true)
end
