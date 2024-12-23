function occupancy_model(sol::LPSolver, mdp::SparseTabularMDP)
    (;T,R,discount) = mdp
    γ = discount
    ns, na = size(R)
    model = Model(sol.sol)
    set_attributes(
        model, 
        kwargs2attrs(sol.settings, solver_defaults(sol.sol))...
    )
    @variable(model, μ[1:ns, 1:na] .≥ 0)
    @expression(model, μs, vec(sum(μ, dims=2)))
    @constraint(model, μs .== (1-γ) .* v0 .+ γ .* sum(T[a]' * μ[:, a] for a ∈ 1:na))
    @objective(model, Max, sum(μ .* R))
    return model
end
