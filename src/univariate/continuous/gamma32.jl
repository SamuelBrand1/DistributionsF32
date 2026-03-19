"""
    GammaF32(α, θ)

Alias for `Distributions.Gamma`. Construct with `Float32` arguments to get
`Float32` semantics for `rand`, `logpdf`, `pdf`, `cdf`, etc.

```julia
GammaF32(1.0f0, 2.0f0)   # Gamma{Float32} with shape 1 and scale 2
GammaF32(2.0f0)           # Gamma{Float32} with shape 2 and unit scale
GammaF32()                # Gamma{Float32}(1, 1)
```

See also: [`Distributions.Gamma`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Gamma)
"""
const GammaF32 = Gamma
GammaF32() = GammaF32(1.0f0, 1.0f0)

Distributions.minimum(::GammaF32) = 0.0f0
Distributions.maximum(::GammaF32) = Inf32
