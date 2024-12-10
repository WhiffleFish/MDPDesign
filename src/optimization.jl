struct RegularizedMDPOptimizationProblem{M}
    pmdp::M
    s_idx::Int64 # replace with μ0
    w::Float64
end

function value_and_grad(p::RegularizedMDPOptimizationProblem, θ)
    (;s_idx, w, pmdp) = p
    frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(pmdp), θ)
    mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
    V = mdp_sol.V[s_idx]
    ∇V = getindex.(MDPDesign.parameterization_gradient(mdp_sol),s_idx)
    l = -V + 0.5 * w * dot(θ,θ)
    ∇l = -∇V .+ w .* θ
    return l, ∇l
end
