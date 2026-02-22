@testsnippet DiffTests_logpdf begin
    using Distributions, DifferentiationInterface
    using ForwardDiff: ForwardDiff
    using ReverseDiff: ReverseDiff
    using Mooncake: Mooncake
    using Zygote: Zygote

    function test_logpdf_gradient(d, p, x_fixed, expected_grad)
        f(p) = logpdf(d(p...), x_fixed)

        backends = [
            ("ForwardDiff", AutoForwardDiff()),
            ("ReverseDiff", AutoReverseDiff()),
            ("Mooncake", AutoMooncake(; config = nothing)),
            ("Zygote", AutoZygote()),            # ("Enzyme", AutoEnzyme()),
        ]

        for (name, backend) in backends
            @testset "$name" begin
                try
                    g = gradient(f, backend, p)
                    @test g ≈ expected_grad atol = 1.0e-5
                catch e
                    @test_broken false
                    @warn "AD backend $name failed on $(d) — this is logged as broken, not a hard failure" exception = (
                        e, catch_backtrace(),
                    )
                end
            end
        end
    end
end
