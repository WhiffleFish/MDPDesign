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
    @expression(model, expected_value, sum(μ .* R))
    @constraint(model, μs .== (1-γ) .* v0 .+ γ .* sum(T[a]' * μ[:, a] for a ∈ 1:na))
    @objective(model, Max, expected_value)
    return model
end

struct ConstrainedModel{M,C}
    model::M
    constraints::C
end

function constrained_model(sol::LPSolver, mdp::TabularCMDP; v0=mdp.initialstate)
    (;T,R,C,constraints,discount) = mdp
    γ = discount
    ns, na = size(R)
    model = Model(sol.sol)
    set_attributes(
        model, 
        kwargs2attrs(sol.settings, solver_defaults(sol.sol))...
    )
    @variable(model, μ[1:ns, 1:na] .≥ 0)
    @expression(model, μs, vec(sum(μ, dims=2)))
    @expression(model, expected_value, sum(μ .* R))
    @constraint(model, μs .== (1-γ) .* v0 .+ γ .* sum(T[a] * μ[:, a] for a ∈ 1:na))
    cost_constraints = map(axes(C, 3)) do i
        @constraint(model, sum(μ .* C[:,:,i]) ≤ constraints[i]) 
    end
    @objective(model, Max, expected_value)
    return ConstrainedModel(model, cost_constraints)
end

struct OptimizedConstrainedModel{M,MT,RT,CT}
    model::M
    μ::MT
    R::RT
    C::CT
end

optimize(m::ConstrainedModel) = optimize!(m)

function JuMP.optimize!(m::ConstrainedModel)
    optimize!(m.model)
    return OptimizedConstrainedModel(
        m.model, 
        JuMP.value.(m.model[:μ]), 
        JuMP.value(m.model[:expected_value]),
        JuMP.value.(m.constraints)
    )
end
