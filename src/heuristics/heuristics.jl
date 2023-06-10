
export Dataproblem, Reportproblem, distance, random_route, generate_instance, cost, model
export nearest_neighbor, clark_wright, two_opt, rms, angular, ils, simulated_annealing, vns

module Heuristic

include("../utils/utils.jl")

using Plots, Random, Coluna, JuMP, Cbc, HiGHS, BlockDecomposition, PlotThemes; theme(:juno)
using Base.Threads: @spawn

#==========================================================================================#
# MODELO MATEMÁTICO                                                                        #
#==========================================================================================#

# esse método gera o modelo MTZ para o problema do caixeiro viajante
function model(data::Dataproblem, opt::Solverparameters, ativo::Bool)

    # construção do modelo
    tsp = Model(Cbc.Optimizer)

    # setando atributos do solver
    # set_optimizer_attribute(tsp, "presolve", "choose")
    # set_optimizer_attribute(tsp, "time_limit", opt.timelimit)
    # set_optimizer_attribute(tsp, "parallel", "on")
    # set_optimizer_attribute(tsp, "threads", opt.threads)
    # set_optimizer_attribute(tsp, "solver", "choose")

    # tsp = Model(
    #     optimizer_with_attributes(
    #         HiGHS.Optimizer,
    #         "seconds"      => opt.timelimit, 
    #         "allowableGap" => opt.gap, 
    #         "Threads"      => opt.threads
    #     )
    # )

    # setando as dimensões
    n = copy(data.n)

    # variáveis
    @variable(tsp, x[i = 1:n, j = 1:n], Bin)
    @variable(tsp, u[i = 1:n] >= 0)

    # função objetivo
    @objective(tsp, Min, sum(data.distance[i,j] * x[i,j] for i in 1:n, j in 1:n))

    # construção das restrições
    @constraint(tsp, [j = 1:n], sum(x[i,j] for i = 1:n if i != j) == 1)
    @constraint(tsp, [i = 1:n], sum(x[i,j] for j = 1:n if i != j) == 1)
    @constraint(tsp, u[1] == 1)
    @constraint(tsp, [i = 1:n, j = 2:n; i ≠ j], u[j] ≥ u[i] + 1 - n * (1-x[i,j]))
    
    # resolvendo o modelo
    optimize!(tsp)

    # validando o status da solução
    if termination_status(tsp) == MOI.OPTIMAL
        x = value.(x)
        rota = tour(x)
        custo = objective_value(tsp)
        nome = "TSP: MODEL-MTZ"
        resultado = Reportproblem(nome, custo, rota)
        output(resultado)
        graph(data, resultado, ativo)
        return resultado
    elseif termination_status(tsp) == MOI.TIME_LIMIT && has_values(tsp)
        x = value.(x)
        rota = tour(x)
        custo = objective_value(tsp)
        nome = "TSP: MODEL-MTZ"
        resultado = Reportproblem(nome, custo, rota)
        output(resultado)
        graph(data, resultado, ativo)
        return resultado
    else
        println("Solução não encontrada !")
    end
end

#==========================================================================================#
# HEURÍSTICAS CONSTRUTIVAS                                                                 #
#==========================================================================================#

# método de construção aleatória de rota
function random_route(data::Dataproblem, ativo::Bool)
    rota::AbstractVector{Integer} = shuffle([1:data.n...])
    nome::String  = "random route"
    custo::Float64 = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

# método do vizinho mais próximo
function nearest_neighbor(data::Dataproblem, ativo::Bool)
    rota::AbstractVector{Integer} = []
    matriz = copy(data.distance)
    j = 1
    k = 1
    @inbounds for i = 1:data.n
        k = argmin(matriz[j,:])
        matriz[j,k]  = 1000.0
        matriz[:,j] .= 1000.0 
        j = k
        push!(rota, k)
    end
    nome::String = "Heurística : nearest neighbor"
    custo::Float64 = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

# método de alocação por ângulos (angular fit)
function angular(data::Data, ativo::Bool)
    
    # quando x < 0, calcula o arco tangente (y / x) + 180
    # quando y < 0, calcula o arco tangente (y / x) + 360

    angulos = []
    
    # trazendo as coordenadas para o centro do plano cartesiano
    coord_x = data.cx .- data.cx[1]
    coord_y = data.cy .- data.cy[1]
    
    # calculando os angulos entre os pontos
    @inbounds for i = 1:data.n
        if coord_x[i] > 0.0 && coord_y[i] > 0.0
            push!(angulos, atand(coord_y[i] / coord_x[i]))
        elseif coord_x[i] > 0.0 && coord_y[i] < 0.0
            push!(angulos, atand(coord_y[i] / coord_x[i]) + 360)
        else
            push!(angulos, atand(coord_y[i] / coord_x[i]) + 180)
        end
    end
    
    rota::AbstractVector{Integer} = [1:data.n...]
    df = [rota angulos]
    rota = sortslices(df, dims=1, by=x->(x[2]), rev=false)[:,1]
    nome  = "Heurística : angular"
    custo = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

# método de cálculo de savings e alocação de clientes
function clark_wright(data::Dataproblem, ativo::Bool)

    load = zeros(Int, data.n)
    rotas = []
    capacidade = 0.0
    # criando as rotas triviais: indo do depósito até um cliente
    for i = 2:data.n
        push!(rotas, [1, i])
    end

    # calculando os savings
    s = savings(data)
    
    # concatenando rotas
    while true
        argmax = nothing
        maxval = 0
        @inbounds for (k, rk) in enumerate(rotas)
            @inbounds for (l, rl) in enumerate(rotas)
                if k != l && maxval < s[rk[end], rl[2]] && load[k] + load[l] <= capacidade
                    argmax = [k, l]
                    maxval = s[rk[end], rl[2]]
                end
            end
        end
        if typeof(argmax) != Nothing
            k, l = argmax
            rotas[k] = [rotas[k]; rotas[l][2:end]]
            load[k] += load[l]
            deleteat!(rotas, l)
            deleteat!(load, l)
        else
            break
        end
    end
    rota::AbstractVector{Integer} = rotas[1]
    nome::String   = "Heuristic: clark and wright"
    custo::Float64 = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

#==========================================================================================#
# HEURÍSTICAS DE MELHORIA                                                                  #
#==========================================================================================#

function two_opt(data::Dataproblem, resultado::Reportproblem, ativo::Bool)

    rota = copy(resultado.rota)
    
    # size checks
    n = copy(data.n)
      
    # how much must each swap improve the cost?
    thresh = 0.02

    # main loop
    # check every possible switch until no 2-swaps reduce objective
    # if the path passed in is a loop (first/last nodes are the same)
    # then we must keep these the endpoints of the path the same
    # ie just keep it a loop, and therefore it doesn't matter which node is at the end
    # if the path is not a cycle, we should respect the endpoints
    switchLow = 2
    switchHigh = n - 1
    need_to_loop = true # always look for swaps at least once
    while need_to_loop
        need_to_loop = false
        # we can't change the first
        @inbounds for i in switchLow:(switchHigh-1)
            @inbounds for j in switchHigh:-1:(i+1)
                cost_change = prv(data.distance, rota, i, j)
                if cost_change + thresh <= 0
                    need_to_loop = true
                    reverse!(rota, i, j)
                end
            end
        end
    end
    nome = "Heuristic : 2-opt"
    custo = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

#==========================================================================================#
# METAHEURÍSTICAS                                                                          #
#==========================================================================================#

# SIMULATED ANNEALING
function simulated_annealing(data::Data, resultado::Reportproblem, temp::Int, iter::Int)
    rota = resultado.rota
    best = resultado
    for i = 1:iter
        new = two_opt(data, resultado, false)
        delta = new.custo - resultado.custo
        if delta < 0
            resultado = new
        elseif rand() < exp(-delta / temp)
            resultado = new
        end
        if resultado.custo < best.custo
            best = resultado
        end
    end
    rota = best.rota
    nome = "Metaheuristic : Simulated Annealing"
    custo = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end
    
# VARIABLE NEIGHBORHOOD SEARCH
# 
function vnd(data::Dataproblem, resultado::Reportproblem, ativo::Bool)
    imp  = true
    best = resultado
    while imp
        imp = false
        if imp != true
            best = two_opt(data, resultado, false)
            if best.custo < resultado.custo
                resultado = best
                imp = true
            end
        end
    end
    rota = best.rota
    nome = "Metaheuristic : VND"
    custo = cost(data, rota)
    resultado = Reportproblem(nome, custo, rota)
    output(resultado)
    graph(data, resultado, ativo)
    return resultado
end

# RANDOM MULTI START
function rms(data::Data, iter::Int, ativo::Bool)
    bestcost = Inf
    bestroute = nothing
    for i = 1:iter
        sol = routedestroy(data)
        current_sol = vnd(data, sol, false)
        if bestcost > current_sol.custo
            bestcost = current_sol.custo
            bestroute = current_sol.rota
        end
    end

    nome  = "Metaheuristic: RMS"
    custo = bestcost
    rota  = bestroute
    resultado = Reportproblem(nome, custo, rota)
    resultado = two_opt(data, resultado, false)
    nome  = "Metaheuristic: RMS"
    custo = resultado.custo
    rota  = resultado.rota
    resultado = Reportproblem(nome, custo, rota)
    graph(data, resultado, ativo)
    return resultado
end

# GRASP - (GREEDY RANDOMIZED ADAPTIVE SEARCH PROCEDURE)
# DESCRIÇÃO DO GRASP: O método GRASP é uma metaheurística que combina elementos de construção de soluções
# e busca local. A construção de soluções é feita de forma gulosa, mas com aleatoriedade controlada,
# enquanto a busca local é usada para melhorar as soluções construídas. O GRASP é um método iterativo
# que, a cada iteração, constrói uma solução gulosa aleatorizada e, em seguida, melhora essa solução
# por meio de uma busca local. O processo é repetido até que um critério de parada seja satisfeito.
function grasp(data::Data, ativo::Bool, iter::Int, alpha::Float64)
    
    # definição das etapas do algoritmo GRASP para o TSP
    # 1. Inicialização
    # 2. Construção da solução
    # 3. Busca local
    # 4. Critério de parada
    # 5. Retorno da melhor solução encontrada

    # método que constroe um solução gulosa aleatorizada
    rota = nearest_neighbor(data, false)

    # método que melhora a solução construída por meio de uma busca local
    resultado = vnd(data, rota, false)

    # método que retorna a melhor solução encontrada
    nome = "Metaheuristic : GRASP"
    custo = resultado.custo
    rota = resultado.rota
    resultado = Reportproblem(nome, custo, rota)
    graph(data, resultado, ativo)
    return resultado
end

# ITERATED LOCAL SEARCH
# DESCRIÇÃO DO ILS: O método ILS é uma metaheurística que combina elementos de construção de soluções
# e busca local. A construção de soluções é feita de forma gulosa, mas com aleatoriedade controlada,
# enquanto a busca local é usada para melhorar as soluções construídas. O ILS é um método iterativo
# que, a cada iteração, constrói uma solução gulosa aleatorizada e, em seguida, melhora essa solução
# por meio de uma busca local. O processo é repetido até que um critério de parada seja satisfeito.
function ils(data::Data, resultado::Reportproblem, iter::Int, ativo::Bool)
    
    # definição das etapas do algoritmo ILS para o TSP
    # 1. Inicialização
    # 2. Construção da solução
    # 3. Busca local
    # 4. Perturbação: perturba a solução encontrada por meio de uma busca local
    # 5. Critério de parada
    # 6. Retorno da melhor solução encontrada

    # método que constroe um solução gulosa aleatorizada
    rota = clark_wright(data, false)

    # método que melhora a solução construída por meio de uma busca local
    resultado = vnd(data, rota, false)

    # método que faz a perturbação da solução
    resultado = rms(data, iter, false)

    nome = "Metaheuristic : ILS"
    custo = resultado.custo
    rota = resultado.rota
    resultado = Reportproblem(nome, custo, rota)
    graph(data, resultado, ativo)
    return resultado
end

# TABU-SEARCH FOR TSP
function tabu_search(data::Data, resultado::Reportproblem)

end

# SCATTER SEARCH
function scatter_search(data::Data, resultado::Reportproblem)
    
end

# GENETIC ALGORITHM
function genetic_algorithm(data::Data, resultado::Reportproblem, ativo::Bool, pop::Int, iter::Int, prob::Float64)
    
    # definição das etapas do algoritmo genético para o TSP
    # 1. Inicialização da população
    # 2. Avaliação da população
    # 3. Seleção dos pais
    # 4. Cruzamento
    # 5. Mutação
    # 6. Avaliação da população
    # 7. Seleção dos sobreviventes
    # 8. Critério de parada
    # 9. Retorno da melhor solução

    function crossover(data, sol1, sol2, prob)
        n = length(sol1.rota)
        rota = zeros(Int, n)
        for i = 1:n
            if rand() < prob
                rota[i] = sol1.rota[i]
            else
                rota[i] = sol2.rota[i]
            end
        end
        rota = vnd(data, rota, false)
        return rota
    end

    # método para mutação, com base na probabilidade
    function mutation(data, rota1, rota2, prob)
        # difinindo a probabilidade de mutação
        n = length(rota1)
        for i = 1:n
            if rand() < prob
                rota1[i] = rota2[i]
            end
        end

        # aplicando o VND
        rota1 = vnd(data, rota1, false)
        return rota1
    end


    # 1. Inicialização da população
    população = []
    for i = 1:pop
        sol = routedestroy(data)
        push!(população, sol)
    end

    # 2. Avaliação da população
    for i = 1:pop
        população[i] = vnd(data, população[i], false)
    end

    # 3. Seleção dos pais, com base na roleta
    pais = []
    for i = 1:pop
        for j = 1:pop
            if i != j
                if população[i].custo < população[j].custo
                    push!(pais, população[i])
                else
                    push!(pais, população[j])
                end
            end
        end
    end

    # 4. Cruzamento
    filhos = []
    for i = 1:pop
        for j = 1:pop
            if i != j
                filho = crossover(data, pais[i], pais[j], prob)
                push!(filhos, filho)
            end
        end
    end

    # 5. Mutação, troca de parte da rotas
    for i = 1:pop
        for j = 1:pop
            if i != j
                filho = mutation(data, filhos[i], filhos[j], prob)
                push!(filhos, filho)
            end
        end
    end

    # 6. Avaliação da população
    for i = 1:pop
        filhos[i] = vnd(data, filhos[i], false)
    end

    # 7. Seleção dos sobreviventes
    sobreviventes = []
    for i = 1:pop
        for j = 1:pop
            if i != j
                if filhos[i].custo < filhos[j].custo
                    push!(sobreviventes, filhos[i])
                else
                    push!(sobreviventes, filhos[j])
                end
            end
        end
    end

    # 8. Critério de parada
    for i = 1:pop
        if sobreviventes[i].custo < resultado.custo
            resultado = sobreviventes[i]
        end
    end

    # 9. Retorno da melhor solução
    nome = "Genetic Algorithm"
    custo = resultado.custo
    rota = resultado.rota
    resultado = Reportproblem(nome, custo, rota)
    graph(data, resultado, ativo)
    return resultado
end

end # end module
