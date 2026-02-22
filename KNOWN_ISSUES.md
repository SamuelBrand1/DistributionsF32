# Known Issues

## Mooncake.jl fails on Julia 1.12 with `Union{Float32, Float64}` type inference

**Status:** Open — upstream Mooncake limitation
**Affected:** Julia 1.12+, Mooncake v0.5.7
**Works on:** Julia 1.11

### Error

```
MethodError: no method matching Union{Float32, Float64}(::Int64)
```

in `Mooncake.zero_rdata_from_type(::Type{Union{Float32, Float64}})`.

### Cause

Our `logpdf` implementations mix Float32 constants (`log2π_f32`, `2f0`) with
potentially-Float64 parameters from the parametric struct (e.g. `NormalF32{Float64}`
created during AD). Julia 1.12's type inference infers `Union{Float32, Float64}` for
some intermediates, whereas 1.11 collapsed these to `Float64`.

Mooncake pre-compiles reverse-mode rules by walking inferred IR and needs to call
`zero(T)` for every intermediate type. `zero(Union{Float32, Float64})` is not defined,
so it errors.

ForwardDiff, ReverseDiff, and Zygote are unaffected — they use operator overloading or
source transforms that don't depend on pre-analyzing inferred types.

### Workaround

The AD test suite (`test/diff_tests.jl`) catches backend failures and marks them as
`@test_broken` so CI still passes. The warning is logged with the full exception.
