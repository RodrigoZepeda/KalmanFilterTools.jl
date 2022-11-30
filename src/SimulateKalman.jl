@doc raw"""
SimulateKalman()

Simulate an implementation of the Kalman Filter for a time period `t` given by:
```math
\begin{aligned}
    x_{k+1} &= A x_k + B u_k + w_k \\
    y_k &= C x_k + D u_k + v_k
\end{aligned}
A \in \matbb{R}^{n\times n}, B \in \mathbb{R}^{n \times m},
C \in \mathbb{R}^{q \times n}, D \in \mathbb{R}^{q \times m},
u \in \mathbb{R}^{m \times t}
```
and where 
```math
x_0\in\mathbb{R}^n
```

Examples:
```julia-repl
julia> #You can only specify the number of simulations
julia> x,y = SimulateKalman(100)
julia> plot(1:size(y,2), y')
julia> #Or the inner dimensions
julia> x,y = SimulateKalman(100, n = 4, m = 3, q = 5)
julia> plot(1:size(y,2), y')
julia> #Or the transition matrices
julia> A = [[0.20, 0.2] [0.5, 0.8]]
julia> B = [[0.01, 0.5, 0.7] [0.8, 0.1, 0.2]]'
julia> C = [[0.71, 0.3] [0.1, 0.3] [0.08, 0.5] [0.2, 0.21]]'
julia> D = [[0.4, 0.1, 0.7] [0.3, 0.25, 0.01] [0.0, 0.1, 0.0] [0.2, 0.37, 0.01]]'
julia> x,y = SimulateKalman(1000, A, B, C, D)
julia> plot(1:size(y,2), y')
```
"""
function SimulateKalman(t::Integer,
    A::AbstractMatrix{<:Real}, 
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real};
    u::Union{AbstractMatrix{<:Real},Nothing}  = nothing,
    x0::Union{AbstractVector{<:Real},Nothing} = nothing,
    DistributionW::UnivariateDistribution = Normal(0,1),
    DistributionV::UnivariateDistribution = Normal(0,1),
) 

    #Check t >= 1
    if (t < 1)
        error("Variable t is $t. Must be an integer >= 1")
    end

    #Check all the values
    n = size(A,1);
    check_matrix_SimulateKalman(A,n,n,"A");
    
    m = size(B,2);
    check_matrix_SimulateKalman(B,n,m,"B");
    
    q = size(C,1);
    check_matrix_SimulateKalman(C,q,n,"C");

    check_matrix_SimulateKalman(D,q,m,"D");

    return simulate_kalman_steps(t, A, B, C, D, u, x0, DistributionW, DistributionV);
    
end

function SimulateKalman(t::Integer;
    n::Integer=4,
    m::Integer=1,
    q::Integer=5,
    u::Union{AbstractMatrix{<:Real},Nothing}  = nothing,
    x0::Union{AbstractVector{<:Real},Nothing} = nothing,
    DistributionW::UnivariateDistribution = Normal(0,1),
    DistributionV::UnivariateDistribution = Normal(0,1),
    force_stable=true,
) 

    if (t < 1 | m < 1 | q < 1)
        error("Values t, m, q must be >= 1")
    end

    #Simulate A matrix (stable)
    if (force_stable)
        Λ = LinearAlgebra.diagm(Random.rand(Beta(0.5, 0.5), n));
    else 
        Λ = LinearAlgebra.diagm(Random.rand(Normal(0.0, 1.0), n));
    end
    Q, R  = LinearAlgebra.qr(Random.rand(Normal(0.0, 1.0), n, n));
    A     = Q*Λ*inv(Q)

    #Simulate B, C and D matrices
    B     = Random.rand(Normal(0.0, 1.0), n, m);
    C     = Random.rand(Normal(0.0, 1.0), q, n);
    D     = Random.rand(Normal(0.0, 1.0), q, m);

    return simulate_kalman_steps(t, A, B, C, D, u, x0, DistributionW, DistributionV);

end

export SimulateKalman

@doc raw"""
Function to simulate the KalmanSteps given the initial values and the matrices
"""
function simulate_kalman_steps(t::Integer,
    A::AbstractMatrix{<:Real}, 
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real},
    u::Union{AbstractMatrix{<:Real},Nothing},
    x0::Union{AbstractVector{<:Real},Nothing},
    DistributionW::UnivariateDistribution,
    DistributionV::UnivariateDistribution)

    #Check the optional vector inputs
    u      = check_u_input_SimulateKalman(u,size(B,2), t);
    x0init = check_x0_SimulateKalman(x0,size(A,1));
    xmat, ymat = kalman_steps(x0init, u, A, B, C, D; DistributionW, DistributionV);

    return xmat, ymat, u, A, B, C, D 
end

function kalman_steps(
    x0::AbstractVector{<:Real},
    u::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real}, 
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real};
    DistributionW::Union{UnivariateDistribution,Nothing}=nothing,
    DistributionV::Union{UnivariateDistribution,Nothing}=nothing)

    #Check the optional vector inputs
    n         = size(A,1)
    t         = size(u,2)
    q         = size(C,1);
    ymat      = Array{Float64}(undef, q, t)
    xmat      = Array{Float64}(undef, n, t + 1)
    xmat[:,1] = x0;

    #Initial values for w and 
    w = zeros(Float64, n, t)
    if (!isnothing(DistributionW))
        for k in 1:t
            Random.rand!(DistributionW, w[:,k])
        end
    end

    v = zeros(Float64, q, t)
    if (!isnothing(DistributionV))
        for k in 1:t
            Random.rand!(DistributionV, v[:,k])
        end
    end

    for k in 1:t
        xmat[:,k + 1], ymat[:,k] = kalman_kth_steps(xmat[:,k], u[:,k], A, B, C, D, w[:,k], v[:,k]);
    end

    return xmat, ymat;
end


function kalman_kth_steps(
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    A::AbstractMatrix{<:Real}, 
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real},
    w::AbstractVector{<:Real},
    v::AbstractVector{<:Real},)

    xnew = A*x + B*u + w;
    ynew = C*x + D*u + v;
    
    return xnew, ynew;
end

function kalman_kth_steps(
    x::AbstractVector,
    u::AbstractVector{<:Real},
    A::AbstractMatrix{<:Real}, 
    B::AbstractMatrix{<:Real},
    C::AbstractMatrix{<:Real},
    D::AbstractMatrix{<:Real})

    xnew = A*x + B*u;
    ynew = C*x + D*u;
    
    return xnew, ynew;
end
export kalman_kth_steps

@doc raw"""
Function to check the u input of SimulateKalman function
"""
function check_u_input_SimulateKalman(u::Union{AbstractMatrix{<:Real},Nothing},m::Integer,t::Integer;eta::Real=0.5)
    if !isnothing(u)
        if ((size(u,1) != m) | (size(u,2) != t))
            error("u must be a $m x $t matrix")
        end
    else 
       u  = Random.rand(Normal(0,1), m, t)
    end
    return u
end

@doc raw"""
Function to check the x0 input of SimulateKalman function
"""
function check_x0_SimulateKalman(x0::Union{AbstractVector{<:Real},Nothing},n::Integer)
    if !isnothing(x0)
        if (size(x0,1) != n)
            error("x0 must be a vector of size $n")
        end
    else
        x0 = Random.rand(Normal(0,1),n)
    end
    return x0
end

@doc raw"""
Function to check matrix dimensions of SimulateKalman function
"""
function check_matrix_SimulateKalman(Mat::AbstractMatrix{<:Real},n::Integer,m::Integer,matname::String)
    if ((size(Mat,1) != n) | (size(Mat,2) != m))
        error("$matname must be a $n x $m matrix")
    end
end