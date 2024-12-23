struct OptimizedMDPModel{M<:Model,C,F<:FrozenParameterizedTabularMDP}
    model::M
    V::Vector{Float64}
    constraints::C
    mdp::F
end

function OptimizedMDPModel(mdp::FrozenParameterizedTabularMDP; kwargs...)
    V, (;model, constraints) = solve_info(LPSolver(;kwargs...), SparseTabularMDP(mdp))
    return OptimizedMDPModel(model, V, constraints, mdp)
end

POMDPs.discount(pmdp::OptimizedMDPModel) = discount(pmdp.mdp)

function parameterization_gradient(m::OptimizedMDPModel)
    (; constraints, V, mdp) = m
    (;DT) = mdp
    λ = mapreduce(vcat, constraints) do c
        dual.(c)
    end
    return map(DT) do ΔT
        _parameterization_gradient(m, ΔT, λ)
    end
end

# for just a single parameter
function _parameterization_gradient(m::OptimizedMDPModel, ∇T, λ)
    (; V, mdp) = m
    (; T, R) = mdp
    na = length(T)
    ns = length(mdp.μ0)
    x = V
    γ = discount(mdp)
    A = mapreduce(hcat, T) do T_i
        I - γ*T_i
    end
    b = mapreduce(hcat, ∇T) do ∇T_i
        -γ * ∇T_i'
    end * λ
    dλdθ =  A \ b
    Ax = mapreduce(vcat, 1:na) do i
        idxs = (ns * (i - 1) + 1) : ns * i
        λa = λ[idxs]
        spdiagm(λa) * (I - γ * T[i])
    end
    bx = mapreduce(vcat, 1:na) do i
        idxs = (ns * (i - 1) + 1) : ns * i
        spdiagm(dλdθ[idxs]) * (-x + R[:, i] + γ * T[i]*x) + spdiagm(λ[idxs]) * γ * ∇T[i] * x
    end
    dxdθ = Ax \ bx
    return dxdθ
end

function parameterization_gradient(T_f, dT_f, R, γ, θ)
    T = T_f(θ)
    ns = size(first(T), 1)
    na = length(T)
    ∇T = dT_f(θ)
    model = Model(HiGHS.Optimizer)
    @variable(model, V[1:ns])
    constraints = map(1:na) do i
        @constraint(model,  -V + R[:,i] .+ γ .* T[i] * V .≤ 0)
    end
    @objective(model, Min, sum(V))
    optimize!(model)
    
    λ = mapreduce(vcat, constraints) do c
        dual.(c)
    end
    A = mapreduce(hcat, T) do T_i
        I - γ*T_i
    end
    b = mapreduce(hcat, ∇T) do ∇T_i
        -γ * ∇T_i'
    end * λ
    dλdθ =  A \ b
    x = JuMP.value.(V)
    Ax = mapreduce(vcat, 1:na) do i
        idxs = (ns * (i - 1) + 1) : ns * i
        λa = λ[idxs]
        diagm(λa) * (I - γ * T[i])
    end
    bx = mapreduce(vcat, 1:na) do i
        idxs = (ns * (i - 1) + 1) : ns * i
        diagm(dλdθ[idxs]) * (-x + R[:, i] + γ * T[i]*x) + diagm(λ[idxs]) * γ * ∇T[i] * x
    end
    dxdθ = Ax \ bx
    return dxdθ, dλdθ
end
