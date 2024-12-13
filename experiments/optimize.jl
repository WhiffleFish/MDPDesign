using POMDPs
using POMDPTools
using POMDPModels
using MDPDesign
using StaticArrays
using MDPDesign.ParameterizedModels
using LinearAlgebra
using Plots
using SparseArrays
using ProgressMeter
using LaTeXStrings
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

gw = FullyParameterizedGridWorld(discount=0.90)
μ0 = zeros(length(states(gw.mdp)))#normalize(ones(length(states(gw.mdp))))
μ0[stateindex(gw.mdp, SA[3,8])] = 1/3
μ0[stateindex(gw.mdp, SA[3,7])] = 1/3
μ0[stateindex(gw.mdp, SA[3,6])] = 1/3
p = RegularizedMDPOptimizationProblem(gw, sparse(μ0), 1e-2)
θ = ones(prod(gw.mdp.size)) * (-3)#zeros(length(states(gw.mdp)))
l, ∇l = value_and_grad(p,θ)

l_hist = [l]
∇l_hist = [∇l]
θ_hist = [θ]
max_iter = 200
α = 1.0
@showprogress for i in 1:max_iter
    θ = θ .- α .* ∇l
    l, ∇l = value_and_grad(p,θ)
    push!(θ_hist, θ)
    push!(l_hist, l)
    push!(∇l_hist, ∇l)
end

p_loss = plot(l_hist, lw=2, ylabel=L"J(\theta)")
plot(map(norm,∇l_hist))

plot(norm.(θ_hist))

θ_norms =  map(θ_hist) do θv
    norm(θv, 2)
end
θt_norms =  map(θ_hist) do θv
    norm(ParameterizedModels.sigmoid.(θv), 2)
end
p_norm = plot(θ_norms, ylabel=L"||\theta||_2", lw=2)

V_hist = map(θ_hist) do θ_i
    frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(gw), θ_i)
    mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
    mdp_sol.V
end
Vs = map(V_hist) do V
    dot(V, μ0)
end

p_value = plot(Vs, ylabel=L"\mu_0^T V^*", lw=2, xlabel="Optimizer Step")

p_final = plot(p_loss, p_norm, p_value, layout=(3,1), suptitle="Bilevel Optimization Process", size=(500, 800))
savefig(p_final, "bilevel-optimization.pdf")
