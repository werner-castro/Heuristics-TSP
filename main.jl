
include("src/tsp.jl")

import .Tsp.Heuristic as tsp

# instanceando o problema com um número p de pontos
p = 120

@time dados = tsp.generate_instance(p)

######################################################################
#                         solução exata                              # 
######################################################################

opt = tsp.Solverparameters(0.000, 6, 1000.0)

resultado = tsp.model(dados, opt, true)

######################################################################
#                  heurísticas construtivas                          # 
######################################################################

# @time resultado = tsp.random_route(dados, true)

# @time resultado = tsp.angular(dados, true)

# @time resultado = tsp.nearest_neighbor(dados, true)

@time resultado = tsp.clark_wright(dados, true) 

######################################################################
#                  heurísticas de melhoria                           # 
######################################################################

@time resultado = tsp.two_opt(dados, resultado, true)

######################################################################
#                          metaheurísticas                           # 
######################################################################

@time resultado = tsp.rms(dados, 100, true)
