struct ParameterizedMDPWrapper{TT, DTT}
    mdp::MDP
    T::TT
    DT::DTT
end

POMDPs.discount(pmdp::ParameterizedMDPWrapper) = discount(pmdp.mdp)

struct ParameterizedTabularMDP{TT, DTT} <: MDP{Int,Int}
    T::TT # T(θ)
    DT::DTT # DT(θ)
    R::Matrix{Float64}
    μ0::Vector{Float64}
    terminal::Set{Int}
    discount::Float64
end

function ParameterizedTabularMDP(pmdp::ParameterizedMDPWrapper)
    R = POMDPTools.ModelTools.reward_s_a(pmdp.mdp)
    μ0 = Array(POMDPTools.ModelTools.inital_probs_vec(pmdp.mdp))
    ts = POMDPTools.ModelTools.terminal_states_set(pmdp.mdp)
    return ParameterizedTabularMDP(
        θ -> param_transition_matrix_a_s_sp(pmdp, θ), 
        θ -> dparam_transition_matrix_a_s_sp(pmdp, θ),
        R, μ0, ts, discount(pmdp)
    )
end

function POMDPTools.SparseTabularMDP(mdp::ParameterizedTabularMDP, θ)
    return SparseTabularMDP(mdp.T(θ), mdp.R, mdp.μ0, mdp.terminal, mdp.discount)
end

struct FrozenParameterizedTabularMDP{TT, DTT, T} <: MDP{Int,Int}
    T::TT
    DT::DTT
    R::Matrix{Float64}
    μ0::Vector{Float64}
    terminal::Set{Int}
    discount::Float64
    θ::T
end

function FrozenParameterizedTabularMDP(mdp::ParameterizedTabularMDP, θ)
    return FrozenParameterizedTabularMDP(
        mdp.T(θ),
        mdp.DT(θ),
        mdp.R,
        mdp.μ0,
        mdp.terminal,
        mdp.discount,
        θ
    )
end

function POMDPTools.SparseTabularMDP(mdp::FrozenParameterizedTabularMDP)
    return SparseTabularMDP(mdp.T, mdp.R, mdp.μ0, mdp.terminal, mdp.discount)
end

param_transition(mdp::ParameterizedMDPWrapper, s, a, θ) = mdp.T(mdp.mdp, s, a, θ)
dparam_transition(mdp::ParameterizedMDPWrapper, s, a, θ) = mdp.DT(mdp.mdp, s, a, θ)

function param_transition_matrix_a_s_sp(pmdp::ParameterizedMDPWrapper, θ)
    mdp = pmdp.mdp
    # Thanks to zach
    na = length(actions(mdp))
    state_space = states(mdp)
    ns = length(state_space)
    transmat_row_A = [Int64[] for _ in 1:na]
    transmat_col_A = [Int64[] for _ in 1:na]
    transmat_data_A = [Float64[] for _ in 1:na]

    for s in state_space
        si = stateindex(mdp, s)
        for a in actions(mdp, s)
            ai = actionindex(mdp, a)
            if isterminal(mdp, s) # if terminal, there is a probability of 1 of staying in that state
                push!(transmat_row_A[ai], si)
                push!(transmat_col_A[ai], si)
                push!(transmat_data_A[ai], 1.0)
            else
                td = param_transition(pmdp, s, a, θ)
                for (sp, p) in weighted_iterator(td)
                    if p > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], si)
                        push!(transmat_col_A[ai], spi)
                        push!(transmat_data_A[ai], p)
                    end
                end
            end
        end
    end
    transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], ns, ns) for a in 1:na]
    # if an action is not valid from a state, the transition is 0.0 everywhere
    # @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(ns)) for a in 1:na) "Transition probabilities must sum to 1"
    return transmats_A_S_S2
end

function dparam_transition_matrix_a_s_sp(pmdp::ParameterizedMDPWrapper, θ)
    mdp = pmdp.mdp
    # Thanks to zach
    na = length(actions(mdp))
    state_space = states(mdp)
    ns = length(state_space)
    transmat_row_A = [Int64[] for _ in 1:na]
    transmat_col_A = [Int64[] for _ in 1:na]
    transmat_data_A = [Float64[] for _ in 1:na]

    for s in state_space
        si = stateindex(mdp, s)
        for a in actions(mdp, s)
            ai = actionindex(mdp, a)
            if isterminal(mdp, s) # if terminal, there is a probability of 1 of staying in that state
                # push!(transmat_row_A[ai], si)
                # push!(transmat_col_A[ai], si)
                # push!(transmat_data_A[ai], 0.0) # parameters cannot affect distribution out of terminal state
                
                # NOTE: parameter gradient is zero here, so we don't add anything! 
                # May come back to bite somehow but idk how 
            else
                td = dparam_transition(pmdp, s, a, θ)
                for (sp, dp) in weighted_iterator(td)
                    if dp > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], si)
                        push!(transmat_col_A[ai], spi)
                        push!(transmat_data_A[ai], dp)
                    end
                end
            end
        end
    end
    transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], ns, ns) for a in 1:na]
    return transmats_A_S_S2
end

function vec_dparam_transition_matrix_a_s_sp(pmdp::ParameterizedMDPWrapper, θv::AbstractArray)
    mdp = pmdp.mdp
    # Thanks to zach
    na = length(actions(mdp))
    state_space = states(mdp)
    ns = length(state_space)
    transmat_row_A = [[Int64[] for _ in 1:na] for _ in eachindex(θv)]
    transmat_col_A = [[Int64[] for _ in 1:na] for _ in eachindex(θv)]
    transmat_data_A = [[Float64[] for _ in 1:na] for _ in eachindex(θv)]

    for s in state_space
        si = stateindex(mdp, s)
        for a in actions(mdp, s)
            ai = actionindex(mdp, a)
            if isterminal(mdp, s) # if terminal, there is a probability of 1 of staying in that state
                # push!(transmat_row_A[ai], si)
                # push!(transmat_col_A[ai], si)
                # push!(transmat_data_A[ai], 0.0) # parameters cannot affect distribution out of terminal state
                
                # NOTE: parameter gradient is zero here, so we don't add anything! 
                # May come back to bite somehow but idk how 
            else
                tdθ = vec_dparam_transition(pmdp, s, a, θv)
                for θ_idx ∈ eachindex(tdθ)
                    td = tdθ[θ_idx]
                    for (sp, dp) in weighted_iterator(td)
                        if dp > 0.0
                            spi = stateindex(mdp, sp)
                            push!(transmat_row_A[θ_idx][ai], si)
                            push!(transmat_col_A[θ_idx][ai], spi)
                            push!(transmat_data_A[θ_idx][ai], dp)
                        end
                    end
                end
            end
        end
    end
    return map(transmat_row_A, transmat_col_A, transmat_data_A) do row,col,data
        [sparse(row[a], col[a], data[a], ns, ns) for a in 1:na]
    end
end
