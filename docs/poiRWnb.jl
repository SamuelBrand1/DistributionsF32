### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ d8537f86-4591-4ee9-8688-a6d576fb005e
using Pkg

# ╔═╡ 470cf6f3-deee-400c-b0aa-3d8748cb516b
Pkg.activate(".")

# ╔═╡ 544505e8-5c4d-471b-8b8d-809c49255655
Pkg.develop(path = "./..")

# ╔═╡ 0270ebe6-cd56-4f83-a53a-2ad9a62e1644
Pkg.add(["Distributions", "Turing", "Plots"])

# ╔═╡ 3f635daf-879f-48e8-94e2-013a4cddcda0
using DistributionsF32, Distributions, Turing, Plots

# ╔═╡ 1e7c4e74-148c-11f1-18e1-1347575e0792
md"
# Trying out Poi RW
"

# ╔═╡ de3d02e8-2675-4421-8b98-f756face82aa
@model function poi_rw(n::Int)
    λ0 ~ Normal(0.0, 10.0)
    σ ~ truncated(Normal(); lower = 0.0)
    ϵt ~ filldist(Normal(), n)

    λt = λ0 .+ σ .* cumsum(ϵt)

    yt ~ arraydist([Poisson(exp(λ)) for λ in λt])

    return yt
end

# ╔═╡ 368e2f95-fe84-44da-b6c0-9c141ff98122
@model function poi_rw_f32(n::Int)
    λ0 ~ NormalF32(0.0, 10.0)
    σ ~ truncated(NormalF32(); lower = 0.0)
    ϵt ~ filldist(NormalF32(), n)

    λt = λ0 .+ σ .* cumsum(ϵt)

    yt ~ arraydist([PoissonF32(exp(λ)) for λ in λt])

    return yt
end

# ╔═╡ a547d972-9f98-497e-9001-680768f9e4e7
n = 30

# ╔═╡ 09d3a619-375c-43ed-adae-32ee17ba665a
mdl = poi_rw(n)

# ╔═╡ 6f5c1329-3d58-44a3-ae68-08ef3b10c042
mdlf32 = poi_rw_f32(n)

# ╔═╡ 0a237b00-a757-4d8a-be24-c517a8885174
begin
    yt1 = mdl()
    yt2 = mdlf32()

    p = plot(; ylabel = "log(counts)", xlabel = "t", title = "$(eltype(yt1)) vs $(eltype(yt2))")
    scatter!(p, yt1 .+ 0.5, yscale = :log10, lab = "poi RW - Distributions.jl")
    scatter!(p, yt2 .+ 0.5, yscale = :log10, lab = "poi RW - DistributionsF32.jl")
    p
end

# ╔═╡ 42e5955b-4628-414d-bd9c-133ba5c28fa6
bad_start_val = 50.0

# ╔═╡ 0880a4da-ee6c-4a02-a8fe-281ac2cb965e
cond_mdl = mdl | (λ0 = bad_start_val,)

# ╔═╡ d0c41fe3-7615-4f5b-a104-8713d317c0d0
cond_mdl_f32 = mdlf32 | (λ0 = Float32(bad_start_val),)

# ╔═╡ fcc3bf0f-d26f-493a-b2fb-0f01f3a82127
try
    cond_mdl()
catch e
    e isa InexactError
end

# ╔═╡ 286d7b00-e509-4142-a984-f0fdcf5cfa44
try
    cond_mdl_f32()
catch e
    e isa InexactError
end

# ╔═╡ 6154bac2-7066-4cea-9276-2fe7343a85ce
true_params = (λ0 = 20.0, σ = 1.0)

# ╔═╡ 72419b8e-6204-4c79-b097-4bec37454692
cond_mdl_for_data = mdl | true_params

# ╔═╡ 7ddc70f8-e290-446f-a0a9-aec27fc9cbcd
yt_data = cond_mdl_for_data()

# ╔═╡ 0c2316fd-fb44-4c01-a419-097312d54583
inference_mdl = mdl | (yt = yt_data,)

# ╔═╡ 08721bbe-55ec-4533-b9c1-5b6766c4bfec
chn = sample(inference_mdl, NUTS(), 1000)

# ╔═╡ 75ce12f2-2975-49e4-9a74-7787aabed358
describe(chn)

# ╔═╡ 44f677ca-0c4e-48a2-b3b9-e707d598eb6c
inference_mdlf32 = mdlf32 | (yt = Float32.(yt_data),)

# ╔═╡ f5c3c38d-56e3-401a-96f4-0bd18d885ba7
chnf32 = sample(inference_mdlf32, NUTS(), 1000)

# ╔═╡ Cell order:
# ╠═1e7c4e74-148c-11f1-18e1-1347575e0792
# ╠═d8537f86-4591-4ee9-8688-a6d576fb005e
# ╠═470cf6f3-deee-400c-b0aa-3d8748cb516b
# ╠═544505e8-5c4d-471b-8b8d-809c49255655
# ╠═0270ebe6-cd56-4f83-a53a-2ad9a62e1644
# ╠═3f635daf-879f-48e8-94e2-013a4cddcda0
# ╠═de3d02e8-2675-4421-8b98-f756face82aa
# ╠═368e2f95-fe84-44da-b6c0-9c141ff98122
# ╠═a547d972-9f98-497e-9001-680768f9e4e7
# ╠═09d3a619-375c-43ed-adae-32ee17ba665a
# ╠═6f5c1329-3d58-44a3-ae68-08ef3b10c042
# ╠═0a237b00-a757-4d8a-be24-c517a8885174
# ╠═42e5955b-4628-414d-bd9c-133ba5c28fa6
# ╠═0880a4da-ee6c-4a02-a8fe-281ac2cb965e
# ╠═d0c41fe3-7615-4f5b-a104-8713d317c0d0
# ╠═fcc3bf0f-d26f-493a-b2fb-0f01f3a82127
# ╠═286d7b00-e509-4142-a984-f0fdcf5cfa44
# ╠═6154bac2-7066-4cea-9276-2fe7343a85ce
# ╠═72419b8e-6204-4c79-b097-4bec37454692
# ╠═7ddc70f8-e290-446f-a0a9-aec27fc9cbcd
# ╠═0c2316fd-fb44-4c01-a419-097312d54583
# ╠═08721bbe-55ec-4533-b9c1-5b6766c4bfec
# ╠═75ce12f2-2975-49e4-9a74-7787aabed358
# ╠═44f677ca-0c4e-48a2-b3b9-e707d598eb6c
# ╠═f5c3c38d-56e3-401a-96f4-0bd18d885ba7
