module KalmanFilterTools

using CSV, DataFrames, LinearAlgebra, Distributions

#Datasets
include("datasets.jl")

#Simulation functions
include("SimulateKalman.jl")

end # module KalmanFilterTools
