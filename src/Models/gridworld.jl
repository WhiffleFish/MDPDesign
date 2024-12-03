function UniformlyParameterizedGridWorld(;kwargs...)
    return ParameterizedMDPWrapper(SimpleGridWorld(;kwargs...), uniform_param_transition, uniform_dparam_transition)
end

function FullyParameterizedGridWorld(;kwargs...)
    return ParameterizedMDPWrapper(SimpleGridWorld(;kwargs...), full_param_transition, full_dparam_transition)
end

function uniform_param_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θ)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = θ # probability of transitioning to the desired cell
        else
            prob = (1.0 - θ)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function uniform_dparam_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θ)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return SparseCat(SA[GWPos(-1,-1)], SA[0.0])
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            dprob = 1.0 # probability of transitioning to the desired cell
        else
            dprob = -1/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += dprob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += dprob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function full_param_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θv::AbstractArray)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end
    s_idx = stateindex(mdp, s)
    θ = θv[s_idx]

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s
    
    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = θ # probability of transitioning to the desired cell
        else
            prob = (1.0 - θ)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function full_dparam_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol, θv::AbstractArray)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return SparseCat(SA[GWPos(-1,-1)], SA[0.0])
    end
    s_idx = stateindex(mdp, s)
    θ = θv[s_idx]

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            dprob = 1.0 # probability of transitioning to the desired cell
        else
            dprob = -1/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += dprob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += dprob
        end
    end

    ∇T_s = convert(Vector{Union{
        Deterministic{Int}, 
        SparseCat{SVector{5, SVector{2, Int64}}, SVector{5, Float64}}}},
        fill(Deterministic(0), prod(mdp.size))
    )

    ∇T_s[s_idx] = SparseCat(convert(SVector, destinations), convert(SVector, probs))

    return ∇T_s
end
