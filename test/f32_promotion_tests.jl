@testsnippet F32PromotionChecks begin
    using ForwardDiff: ForwardDiff

    """
        _contains_float64(T) :: Bool

    Check if a type `T` is or contains `Float64`. Handles:
    - Direct `Float64` match
    - `Union` types (e.g., `Union{Float32, Float64}`)
    - Parameterized types (e.g., `Dual{Tag, Float64, N}`)
    """
    _contains_float64(::Any) = false
    function _contains_float64(T::Type)
        T === Float64 && return true
        if T isa Union
            return _contains_float64(T.a) || _contains_float64(T.b)
        end
        if T isa DataType && !isempty(T.parameters)
            return any(T.parameters) do p
                p isa Type && _contains_float64(p)
            end
        end
        return false
    end

    """
        has_float64_in_ir(f, argtypes::Tuple; optimize=true) :: Bool

    Return `true` if the optimized typed IR for `f(argtypes...)` contains any
    SSA value, slot, or return type that is or contains `Float64`. 

    It does this by looping through the `code_typed` output for `f(argtypes...)` and checking the return type, SSA value types, and slot types for any occurrence of `Float64`.

    This catches hidden Float64 promotions that `@inferred` misses (e.g., Float64
    intermediate computation with a final Float32 conversion).

    Note: `code_typed` only sees Julia-level IR. Promotions inside `ccall`
    (e.g., some SpecialFunctions.jl methods) are opaque to this check.
    """
    function has_float64_in_ir(f, argtypes::Tuple; optimize=true)
        ci_vec = Base.code_typed(f, argtypes; optimize=optimize)
        for (ci, rettype) in ci_vec
            _contains_float64(rettype) && return true
            for ssatype in ci.ssavaluetypes # single static assignment types (local computation results)
                _contains_float64(ssatype) && return true
            end
            if ci.slottypes !== nothing # slot types (stack-allocated variables, e.g., for loops)
                for slottype in ci.slottypes
                    _contains_float64(slottype) && return true
                end
            end
        end
        return false
    end

    """
        find_float64_in_ir(f, argtypes::Tuple; optimize=true) :: Vector

    Like `has_float64_in_ir` but returns a vector of `(location, type)` pairs
    for debugging. `location` is a string like "ssa:5", "slot:2", or "return".
    """
    function find_float64_in_ir(f, argtypes::Tuple; optimize=true)
        results = Tuple{String, Any}[]
        ci_vec = Base.code_typed(f, argtypes; optimize=optimize)
        for (ci, rettype) in ci_vec
            if _contains_float64(rettype)
                push!(results, ("return", rettype))
            end
            for (i, ssatype) in enumerate(ci.ssavaluetypes)
                if _contains_float64(ssatype)
                    push!(results, ("ssa:$i", ssatype))
                end
            end
            if ci.slottypes !== nothing
                for (i, slottype) in enumerate(ci.slottypes)
                    if _contains_float64(slottype)
                        push!(results, ("slot:$i", slottype))
                    end
                end
            end
        end
        return results
    end

    """
        has_float64_in_forwarddiff_gradient_ir(f, n_params::Int) :: Bool

    Check if calling `f` with a `Vector` of ForwardDiff `Dual{Tag, Float32, N}`
    numbers introduces any Float64 in the IR. This tests the gradient computation
    path for Float64 promotions.

    `f` should accept a `Vector{<:Real}` and return a scalar (e.g.,
    `p -> logpdf(NormalF32(p[1], p[2]), 1.5f0)`).
    """
    function has_float64_in_forwarddiff_gradient_ir(f, n_params::Int)
        D = ForwardDiff.Dual{ForwardDiff.Tag{typeof(f), Float32}, Float32, n_params}
        return has_float64_in_ir(f, (Vector{D},))
    end

    # ── Reporting ────────────────────────────────────────────────────

    """
        promotion_report(dist_name, checks; header=true)

    Print a formatted promotion report to the REPL.

    `checks` is a vector of `(label, promotes::Bool, n_f64_sites::Int)` tuples.
    """
    function promotion_report(dist_name::AbstractString, checks::Vector; header::Bool=true)
        if header
            printstyled("┌─ Float64 promotion report: ", bold=true)
            printstyled(dist_name, bold=true, color=:cyan)
            println()
        end

        max_label = maximum(length(first(c)) for c in checks)
        for (label, promotes, n_sites) in checks
            padded = rpad(label, max_label)
            if promotes
                printstyled("│  $padded  ", color=:default)
                printstyled("PROMOTES", color=:red, bold=true)
                printstyled(" ($n_sites site$(n_sites == 1 ? "" : "s") in IR)\n", color=:light_black)
            else
                printstyled("│  $padded  ", color=:default)
                printstyled("clean ✓\n", color=:green)
            end
        end
        printstyled("└─\n", color=:default)
    end

    """
        check_and_report(dist_name, function_checks; header=true)

    Run IR promotion checks and print a report. Returns the checks vector
    for use in `@test` assertions.

    `function_checks` is a vector of `(label, f, argtypes)` tuples.
    """
    function check_and_report(dist_name::AbstractString, function_checks::Vector; header::Bool=true)
        results = Tuple{String, Bool, Int}[]
        for (label, f, argtypes) in function_checks
            sites = find_float64_in_ir(f, argtypes)
            push!(results, (label, !isempty(sites), length(sites)))
        end
        promotion_report(dist_name, results; header=header)
        return results
    end
end
