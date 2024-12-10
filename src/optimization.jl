struct RegularizedMDPOptimizationProblem{M}
    pmdp::M
    μ0::SparseVector{Float64, Int64}
    w::Float64
end

RegularizedMDPOptimizationProblemO(pmdp; μ0=pmdp.μ0, w=1.0) = RegularizedMDPOptimizationProblem(pmdp, μ0, w)

function value_and_grad(p::RegularizedMDPOptimizationProblem, θ)
    (;μ0, w, pmdp) = p
    frozen_mdp = MDPDesign.FrozenParameterizedTabularMDP(MDPDesign.ParameterizedTabularMDP(pmdp), θ)
    mdp_sol = MDPDesign.OptimizedMDPModel(frozen_mdp)
    V = dot(μ0, mdp_sol.V)
    ∇V = map(MDPDesign.parameterization_gradient(mdp_sol)) do dVdθi
        dot(μ0, dVdθi)
    end
    l = -V + 0.5 * w * dot(θ,θ)
    ∇l = -∇V .+ w .* θ
    return l, ∇l
end
