"""
    GammaF32(α, θ)

Convenience constructor that returns `Gamma{Float32}` bypassing `check_args`
(matching NumPyro/JAX semantics). All methods from `Distributions.Gamma` work as-is.

```julia
GammaF32()                # Gamma{Float32}(1, 1)
GammaF32(α)               # Gamma{Float32} with shape α and unit scale
GammaF32(α, θ)            # Gamma{Float32} with shape α and scale θ
```

See also: [`Distributions.Gamma`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Gamma)
"""
GammaF32(α::Real, θ::Real) = Gamma(α, θ; check_args = false)
GammaF32(α::Real) = Gamma(α, 1.0f0; check_args = false)
GammaF32() = Gamma(1.0f0, 1.0f0; check_args = false)
