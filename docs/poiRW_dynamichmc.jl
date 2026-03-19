"""
Poisson Random Walk model using DynamicHMC (Float32-compatible NUTS).

This is the DynamicHMC equivalent of poiRWnb.jl.
Run with: cd docs && julia --project=dynamichmc_env poiRW_dynamichmc.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "dynamichmc_env"))

using DistributionsF32, Distributions
using DynamicHMC, TransformVariables, LogDensityProblems, LogDensityProblemsAD
using ForwardDiff
using Statistics, Random, LinearAlgebra, SpecialFunctions
using Plots

#####
##### LogDensityProblems wrapper for TransformVariables
#####

"""
Wraps a callable log density `f` with a TransformVariables transformation `t`,
implementing the LogDensityProblems API so DynamicHMC can sample from it.

`f(θ)` is the log density on the *constrained* parameter space.
This wrapper handles the unconstrained→constrained transform + Jacobian.
"""
struct TransformedProblem{F,T}
    f::F
    trans::T
end

LogDensityProblems.dimension(p::TransformedProblem) = TransformVariables.dimension(p.trans)
LogDensityProblems.capabilities(::Type{<:TransformedProblem}) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(p::TransformedProblem, x)
    transform_logdensity(p.trans, p.f, x)
end

#####
##### Model definition: Poisson Random Walk
#####

# The transformation from unconstrained ℝᵈ → parameter space
function make_transformation(n)
    as((λ0 = asℝ, σ = asℝ₊, ϵt = as(Array, n)))
end

# Float64 log density
function poi_rw_logdensity(yt)
    function(θ)
        (; λ0, σ, ϵt) = θ

        # Priors
        ℓ = logpdf(Normal(0.0, 10.0), λ0)
        ℓ += logpdf(Normal(), σ)  # half-normal (constraint handled by transform)
        for i in eachindex(ϵt)
            ℓ += logpdf(Normal(), ϵt[i])
        end

        # Likelihood
        λt = λ0 .+ σ .* cumsum(ϵt)
        for i in eachindex(λt)
            ℓ += yt[i] * λt[i] - exp(λt[i]) - lgamma(yt[i] + 1)
        end

        return ℓ
    end
end

# Float32 log density using DistributionsF32
function poi_rw_logdensity_f32(yt)
    yt32 = Float32.(yt)
    function(θ)
        (; λ0, σ, ϵt) = θ

        # Priors — Float32
        ℓ = logpdf(NormalF32(0f0, 10f0), λ0)
        ℓ += logpdf(NormalF32(), σ)
        for i in eachindex(ϵt)
            ℓ += logpdf(NormalF32(), ϵt[i])
        end

        # Likelihood
        λt = λ0 .+ σ .* cumsum(ϵt)
        for i in eachindex(λt)
            ℓ += yt32[i] * λt[i] - exp(λt[i]) - lgamma(yt32[i] + one(yt32[i]))
        end

        return ℓ
    end
end

#####
##### Generate synthetic data
#####

n = 30
true_params = (λ0 = 2.0, σ = 0.3)

Random.seed!(42)
true_ϵt = randn(n)
true_λt = true_params.λ0 .+ true_params.σ .* cumsum(true_ϵt)
yt_data = [rand(Poisson(exp(λ))) for λ in true_λt]

println("Generated $n observations")
println("True λ0 = $(true_params.λ0), true σ = $(true_params.σ)")
println("Observed counts range: $(minimum(yt_data)) to $(maximum(yt_data))")

#####
##### Sample with Float64 DynamicHMC
#####

println("\n--- Float64 DynamicHMC NUTS ---")
trans = make_transformation(n)
P64 = TransformedProblem(poi_rw_logdensity(yt_data), trans)
∇P64 = ADgradient(:ForwardDiff, P64)

rng = Random.default_rng()
results64 = mcmc_with_warmup(rng, ∇P64, 1000; reporter = NoProgressReport())

posterior64 = results64.posterior_matrix
println("Posterior eltype: $(eltype(posterior64))")
println("Posterior size: $(size(posterior64))")

# Extract named parameters
posterior_named64 = [transform(trans, c) for c in eachcol(posterior64)]
λ0_samples64 = [p.λ0 for p in posterior_named64]
σ_samples64 = [p.σ for p in posterior_named64]

println("λ0: mean=$(round(mean(λ0_samples64); digits=3)), std=$(round(std(λ0_samples64); digits=3)) (true: $(true_params.λ0))")
println("σ:  mean=$(round(mean(σ_samples64); digits=3)), std=$(round(std(σ_samples64); digits=3)) (true: $(true_params.σ))")

#####
##### Sample with Float32 DynamicHMC
#####

println("\n--- Float32 DynamicHMC NUTS ---")
P32 = TransformedProblem(poi_rw_logdensity_f32(yt_data), trans)
∇P32 = ADgradient(:ForwardDiff, P32)

# Start from Float32 position
q0_32 = Float32.(randn(TransformVariables.dimension(trans)))

results32 = mcmc_with_warmup(rng, ∇P32, 1000;
    initialization = (q = q0_32,),
    reporter = NoProgressReport())

posterior32 = results32.posterior_matrix
println("Posterior eltype: $(eltype(posterior32))")
println("Posterior size: $(size(posterior32))")

# Extract named parameters
posterior_named32 = [transform(trans, c) for c in eachcol(posterior32)]
λ0_samples32 = [p.λ0 for p in posterior_named32]
σ_samples32 = [p.σ for p in posterior_named32]

println("λ0: mean=$(round(mean(λ0_samples32); digits=3)), std=$(round(std(λ0_samples32); digits=3)) (true: $(true_params.λ0))")
println("σ:  mean=$(round(mean(σ_samples32); digits=3)), std=$(round(std(σ_samples32); digits=3)) (true: $(true_params.σ))")

#####
##### Compare
#####

println("\n--- Comparison ---")
println("Float64 posterior eltype: $(eltype(posterior64))")
println("Float32 posterior eltype: $(eltype(posterior32))")

p1 = scatter(yt_data, label="observed counts", xlabel="t", ylabel="counts",
             title="Poisson Random Walk Data")

p2 = histogram(λ0_samples64, alpha=0.5, label="Float64", title="λ0 posterior",
               normalize=:pdf)
histogram!(p2, λ0_samples32, alpha=0.5, label="Float32", normalize=:pdf)
vline!(p2, [true_params.λ0], label="true", linewidth=2)

p3 = histogram(σ_samples64, alpha=0.5, label="Float64", title="σ posterior",
               normalize=:pdf)
histogram!(p3, σ_samples32, alpha=0.5, label="Float32", normalize=:pdf)
vline!(p3, [true_params.σ], label="true", linewidth=2)

plt = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
savefig(plt, joinpath(@__DIR__, "poiRW_dynamichmc_results.png"))
println("\nPlot saved to docs/poiRW_dynamichmc_results.png")
