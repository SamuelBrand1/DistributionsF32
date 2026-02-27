@testsnippet DiffBackends begin
    using Chairmarks
    using Distributions, DifferentiationInterface, DifferentiationInterfaceTest
    using Enzyme: Enzyme, Reverse, Forward
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake
    using PolyesterForwardDiff: PolyesterForwardDiff
    using ReverseDiff: ReverseDiff
    using Zygote: Zygote

    # Define the backends for testing
    backends = [
        AutoEnzyme(; mode = Reverse),
        AutoEnzyme(; mode = Forward),
        AutoForwardDiff(),
        AutoMooncake(; config = nothing),
        AutoPolyesterForwardDiff(),
        AutoReverseDiff(),
        AutoZygote(),
    ]

    # Define the function to compute logpdf for differentiation
    logpdf_dist(p, dist, x) = logpdf(dist(p...), x)
end
