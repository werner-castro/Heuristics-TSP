
# meta pacote para o problema do tsp

export Heuristic

module Tsp

    include("heuristics/heuristics.jl")

    using .Heuristic

end #  end module