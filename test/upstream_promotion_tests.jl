@testitem "Distributions.jl Float64 promotion audit" setup = [F32PromotionChecks] begin
    using Distributions, Random

    # This test audits upstream Distributions.jl for Float64 promotions.
    # The report prints to REPL so you can see at a glance which functions
    # need custom F32 implementations.
    #
    # If upstream fixes a promoting function → test fails → review whether
    # we can drop our custom implementation.
    # If upstream regresses a clean function → test fails → we need a custom version.

    rng = Random.default_rng()

    # ── Normal{Float32} ──────────────────────────────────────────────
    # As of Distributions v0.25.123, Normal{Float32} is clean for all ops.
    # NormalF32 is still justified for: custom struct, AD-friendly constructors,
    # and guaranteed Float32 semantics independent of upstream changes.
    normal = Normal(1.0f0, 2.0f0)

    r_normal = check_and_report("Distributions.Normal{Float32}", [
        ("logpdf", logpdf, (typeof(normal), Float32)),
        ("pdf",    pdf,    (typeof(normal), Float32)),
        ("cdf",    cdf,    (typeof(normal), Float32)),
        ("rand",   rand,   (typeof(rng), typeof(normal))),
    ])
    for (label, promotes, _) in r_normal
        @test !promotes  # all clean — fails if upstream regresses
    end

    # ── Gamma{Float32} ──────────────────────────────────────────────
    gamma = Gamma(2.0f0, 3.0f0)

    r_gamma = check_and_report("Distributions.Gamma{Float32}", [
        ("logpdf", logpdf, (typeof(gamma), Float32)),
        ("pdf",    pdf,    (typeof(gamma), Float32)),
        ("cdf",    cdf,    (typeof(gamma), Float32)),
        ("rand",   rand,   (typeof(rng), typeof(gamma))),
    ])
    @test !r_gamma[1][2]  # logpdf clean
    @test !r_gamma[2][2]  # pdf clean
    @test r_gamma[3][2]   # cdf promotes — fails if upstream fixes it
    @test r_gamma[4][2]   # rand promotes — fails if upstream fixes it

    # ── Poisson (Float32 params) ────────────────────────────────────
    # Poisson in Distributions.jl uses Float64 params by default.
    poisson = Poisson(5.0f0)

    r_poisson = check_and_report("Distributions.Poisson{Float32}", [
        ("logpdf", logpdf, (typeof(poisson), Float32)),
        ("pdf",    pdf,    (typeof(poisson), Float32)),
        ("cdf",    cdf,    (typeof(poisson), Float32)),
        ("rand",   rand,   (typeof(rng), typeof(poisson))),
    ])
    for (label, promotes, _) in r_poisson
        @test promotes  # all promote — fails if upstream fixes it
    end

    # ── Future candidates ───────────────────────────────────────────
    # Uncomment to triage whether custom F32 implementations are needed

    # exponential = Exponential(2.0f0)
    # check_and_report("Distributions.Exponential{Float32}", [
    #     ("logpdf", logpdf, (typeof(exponential), Float32)),
    #     ("pdf",    pdf,    (typeof(exponential), Float32)),
    #     ("cdf",    cdf,    (typeof(exponential), Float32)),
    #     ("rand",   rand,   (typeof(rng), typeof(exponential))),
    # ])

    # beta = Beta(2.0f0, 3.0f0)
    # check_and_report("Distributions.Beta{Float32}", [
    #     ("logpdf", logpdf, (typeof(beta), Float32)),
    #     ("pdf",    pdf,    (typeof(beta), Float32)),
    #     ("cdf",    cdf,    (typeof(beta), Float32)),
    #     ("rand",   rand,   (typeof(rng), typeof(beta))),
    # ])
end
