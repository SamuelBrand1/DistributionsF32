@testitem "NormalF32 construction" begin
    using Distributions

    # Default constructor
    d = NormalF32()
    @test d.μ === 0.0f0
    @test d.σ === 1.0f0

    # Single arg
    d = NormalF32(2.0f0)
    @test d.μ === 2.0f0
    @test d.σ === 1.0f0

    # Mixed-type inputs get converted to Float32
    d = NormalF32(1.0, 2.0f0)
    @test d.μ === 1.0f0
    @test d.σ === 2.0f0

    # Same-type non-Float32 inputs are preserved (enables AD with Dual numbers)
    d = NormalF32(1.0, 2.0)
    @test d.μ === 1.0
    @test d.σ === 2.0

    # Single-arg always converts to Float32
    d = NormalF32(1.0)
    @test d.μ === 1.0f0
    @test d.σ === 1.0f0

    # GaussianF32 alias
    @test GaussianF32 === NormalF32
end

@testitem "NormalF32 no type promotion — params and statistics" begin
    using Distributions

    d = NormalF32(1.0f0, 2.0f0)

    @test params(d) === (1.0f0, 2.0f0)
    @test partype(d) === Float32
    @test eltype(NormalF32) === Float32

    @test mean(d) isa Float32
    @test median(d) isa Float32
    @test mode(d) isa Float32
    @test var(d) isa Float32
    @test std(d) isa Float32
    @test skewness(d) isa Float32
    @test kurtosis(d) isa Float32
    @test entropy(d) isa Float32

    @test minimum(d) === -Inf32
    @test maximum(d) === Inf32
end

@testitem "NormalF32 no type promotion — evaluation" begin
    using Distributions

    d = NormalF32(0.0f0, 1.0f0)

    # logpdf, pdf, cdf, logcdf should return Float32 when given Float32 input
    @test logpdf(d, 0.0f0) isa Float32
    @test pdf(d, 0.0f0) isa Float32
    @test cdf(d, 0.0f0) isa Float32
    @test logcdf(d, 0.0f0) isa Float32
    @test gradlogpdf(d, 0.0f0) isa Float32

    # Verify known values for standard normal
    @test logpdf(d, 0.0f0) ≈ -Float32(log(2π)) / 2.0f0
    @test pdf(d, 0.0f0) ≈ 1.0f0 / sqrt(2.0f0 * Float32(π))
    @test cdf(d, 0.0f0) ≈ 0.5f0
    @test gradlogpdf(d, 0.0f0) ≈ 0.0f0

    # Non-standard normal
    d2 = NormalF32(3.0f0, 2.0f0)
    @test logpdf(d2, 3.0f0) ≈ -log(2.0f0) - Float32(log(2π)) / 2.0f0
    @test cdf(d2, 3.0f0) ≈ 0.5f0
    @test gradlogpdf(d2, 3.0f0) ≈ 0.0f0
end

@testitem "NormalF32 no type promotion — sampling" begin
    using Distributions, Random

    d = NormalF32(1.0f0, 2.0f0)
    rng = Random.default_rng()

    # Single sample is Float32
    s = rand(rng, d)
    @test s isa Float32

    # rand! fills Float32 array
    A = zeros(Float32, 100)
    rand!(rng, d, A)
    @test eltype(A) === Float32
    @test all(isfinite, A)
end

@testitem "NormalF32 affine transformations" begin
    using Distributions

    d = NormalF32(1.0f0, 2.0f0)

    d2 = d + 3.0f0
    @test d2.μ === 4.0f0
    @test d2.σ === 2.0f0

    d3 = 2.0f0 * d
    @test d3.μ === 2.0f0
    @test d3.σ === 4.0f0

    # Negative scaling
    d4 = -1.0f0 * d
    @test d4.μ === -1.0f0
    @test d4.σ === 2.0f0
end

@testitem "NormalF32 AD: multi-backend gradient of logpdf w.r.t. parameters" setup = [DiffTests_logpdf] begin
    using Distributions

    # Analytical gradients for N(μ=0, σ=1) at x=1.5:
    #   ∂/∂μ logpdf = (x - μ) / σ² = 1.5
    #   ∂/∂σ logpdf = ((x - μ)² - σ²) / σ³ = (2.25 - 1) / 1 = 1.25
    test_logpdf_gradient(NormalF32, [0.0, 1.0], 1.5f0, [1.5, 1.25])
end
