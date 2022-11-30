using KalmanFilterTools, Plots, Random, Turing, LinearAlgebra
Random.seed!(34323);

#First version of Kalman
A = [[0.20, 0.2] [0.5, 0.8]]
B = [[0.01, 0.5, 0.7] [0.8, 0.1, 0.2]]'
C = [[0.71, 0.3] [0.1, 0.3] [0.08, 0.5] [0.2, 0.21]]'
D = [[0.4, 0.1, 0.7] [0.3, 0.25, 0.01] [0.0, 0.1, 0.0] [0.2, 0.37, 0.01]]'
x,y = SimulateKalman(1000, A, B, C, D)

plot(1:size(y,2), y', 
    plot_titlelocation=:left,
    plot_titlefontvalign=:bottom,
    plot_titlevspan=0.2,
    foreground_color_grid = "gray75",
    plot_titlefontfamily="times",
    tickfontfamily="times",
    guidefontfamily="times",
    gridstyle=:dot,
    gridlinewidth=1.0,
    gridalpha=0.5,
    grid=:xy,
    plot_title = "\n\nKalman filter simulations", 
    background_color="#FAF6EC",
    xlabel = "Time\n", 
    ylabel = "\ny(t)",
    legend = false,
    formatter=:plain,
    tick_direction=:out,
    xguidefonthalign = :right, 
    yguidefontvalign = :top,
    palette = :Dark2_5)


#First version of 1000
x,y,u = SimulateKalman(1000, m = 5, q = 5, n = 3)

plot(1:size(y,2), y', 
    plot_titlelocation=:left,
    plot_titlefontvalign=:bottom,
    plot_titlevspan=0.2,
    plot_titlefontfamily="times",
    foreground_color_grid = "gray75",
    gridstyle=:dot,
    gridlinewidth=1.0,
    tickfontfamily="times",
    guidefontfamily="times",
    gridalpha=0.5,
    grid=:xy,
    plot_title = "\n\nKalman filter simulations", 
    background_color="#FAF6EC",
    xlabel = "Time (t)\n", 
    ylabel = "\ny(t)",
    legend = false,
    formatter=:plain,
    tick_direction=:out,
    xguidefonthalign = :right, 
    yguidefontvalign = :top,
    palette = :Dark2_5)


@model function KalmanEstim(
    y::AbstractMatrix{<:Real}; 
    u::Union{AbstractMatrix{<:Real},Nothing} = nothing,
    n::Integer = size(y,1),
    m::Integer = ifelse(!isnothing(u), size(u,1), n),
    q::Integer = size(y,1),
    prior_SigmaA = rand(LKJ(n*n, 0.5)),
    prior_SigmaB = rand(LKJ(n*m, 0.5)),
    prior_SigmaC = rand(LKJ(q*n, 0.5)),
    prior_SigmaD = rand(LKJ(q*m, 0.5)),
    )

    t    = size(y,2);

    if (isnothing(u))
        u = zeros(Float64,m,t)
    end
    
    σₓ    ~ Truncated(Cauchy(0.0, 2.5), 0, Inf)
    σ     ~ Truncated(Cauchy(0.0, 2.5), 0, Inf)

    mu_A  ~ MvNormal(zeros(n*n), Diagonal(repeat([0.01], n*n)))
    mu_B  ~ MvNormal(zeros(n*m), Diagonal(repeat([0.01], n*m)))
    mu_C  ~ MvNormal(zeros(q*n), Diagonal(repeat([0.01], q*n)))
    mu_D  ~ MvNormal(zeros(q*m), Diagonal(repeat([0.01], q*m)))

    Avec  ~ MvNormal(mu_A, prior_SigmaA)
    Bvec  ~ MvNormal(mu_B, prior_SigmaB)
    Cvec  ~ MvNormal(mu_C, prior_SigmaC)
    Dvec  ~ MvNormal(mu_D, prior_SigmaD)

    A = reshape(Avec, n, n)
    B = reshape(Bvec, n, m)
    C = reshape(Cvec, q, n)
    D = reshape(Dvec, q, m)

    #Initial values for w and 
    x0 ~ MvNormal(zeros(Real, n), Diagonal(repeat([σₓ], n)))
    x = x0

    for k in 1:t
        x, ymat = kalman_kth_steps(x, u[:,k], A, B, C, D)
        y[:,k] ~ MvNormal(ymat, Diagonal(repeat([σ], q)))
    end
end

model = KalmanEstim(y, u = u, m = 5, q = 5, n = 3);

nsamples = 2_000

chains   = sample(model, NUTS(), MCMCThreads(), nsamples, 3, drop_warmup=true);
dichains = describe(chains)

#chains = sample(model, Prior(), nsamples);
function TuringKalmanGetParameters(chains::Chains, n::Integer, m::Integer, q::Integer)
    paramlist = MCMCChains.get_params(chains)
    nsims = length(paramlist.lp)
    σ     = vec(paramlist.σ)
    x0    = TuringGetMatrix("x0", n, 1, nsims)
    A     = TuringGetMatrix("Avec", n, n, nsims)
    B     = TuringGetMatrix("Bvec", n, m, nsims)
    C     = TuringGetMatrix("Cvec", q, n, nsims)
    D     = TuringGetMatrix("Dvec", q, m, nsims)
    return A,B,C,D,σ,x0
end

function TuringGetMatrix(matrixname::String, nrows::Integer, ncols::Integer, nsims::Integer)
    Avec  = Array{Float64}(undef, nsims, nrows*ncols)
    Amat  = Array{Matrix{Float64}}(undef, nsims)
    for sim in 1:nsims
        for i in 1:(nrows*ncols)
            Avec[sim,i] = paramlist[Symbol(matrixname)][i][sim]
        end
        Amat[sim] = reshape(Avec[sim,:],nrows,ncols)
    end
    return Amat;
end

A,B,C,D,σ,x0 = TuringKalmanGetParameters(chains, 3 ,5,5)

xmat, yestim = SimulateKalman(size(u,2),A[1],B[1],C[1],D[1];u =u, x0=vec(x0[1]))


plot(1:size(y,2), yestim')