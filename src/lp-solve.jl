struct LPSolver{SOL, ST}
    sol::SOL
    settings::ST
    function LPSolver(sol=HiGHS.Optimizer; kwargs...)
        return new{typeof(sol), typeof(kwargs)}(sol, kwargs)
    end
end

const HiGHS_DEFAULTS = (;log_to_console=false)

solver_defaults(::Any) = (;)
solver_defaults(::Type{HiGHS.Optimizer}) = HiGHS_DEFAULTS

function kwargs2attrs(kwargs)
    return map(keys(kwargs), values(kwargs)) do k,v
        string(k) => v
    end
end

kwargs2attrs(kwargs, defaults) = kwargs2attrs(merge(defaults, kwargs))

function POMDPTools.solve_info(sol::LPSolver, mdp::SparseTabularMDP)
    (;T,R,discount) = mdp
    γ = discount
    ns = length(states(mdp))

    model = Model(sol.sol)
    set_attributes(
        model, 
        kwargs2attrs(sol.settings, solver_defaults(sol.sol))...
    )
    @variable(model, V[1:ns])
    for i ∈ actions(mdp)
        @constraint(model, V .≥ R[:,i] .+ γ .* T[i] * V)
    end
    @objective(model, Min, sum(V))
    optimize!(model)
    return JuMP.value.(V), (; model, constraints)
end

optimal_actions(mdp::SparseTabularMDP, V) = optimal_actions(mdp.R, mdp.T, mdp.discount, V)
optimal_actions(R,T,γ,V) = map(argmax∘tuple, (R[:,i] .+ γ .* T[i] * V for i ∈ actions(mdp))...)

