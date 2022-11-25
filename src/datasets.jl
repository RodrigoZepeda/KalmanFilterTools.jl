#Inspired by https://github.com/JuliaStats/RDatasets.jl
"""
nile()

Measurements of the annual flow of the river Nile at Aswan (formerly Assuan), 
1871–1970, in 10^8 m^3. "with apparent changepoint near 1898” (Cobb(1978), Table 1, p.249). 

The data was obtained from R's `datasets` package.

# Example
```julia-repl
julia> nile()
100×2 DataFrame
 Row │ Year   Flow  
     │ Int64  Int64 
─────┼──────────────
   1 │  1871   1120
   2 │  1872   1160
  ⋮  │   ⋮      ⋮
 100 │  1970    740
     97 rows omitted
```
"""
function nile()
    csvname = joinpath(@__DIR__, "..", "data", "nile.csv")
    if isfile(csvname)
        return CSV.read(csvname, DataFrame)
    end
    error("Unable to locate dataset file nile.csv")
end

export nile