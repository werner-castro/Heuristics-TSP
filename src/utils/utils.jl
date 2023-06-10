
#==========================================================================================#
# ESTRUTURAS DE DADOS                                                                      #
#==========================================================================================#

abstract type Data end

# estrutura com os dados do problema
struct Dataproblem <: Data
    n::Int64                   # número de cidade
    cx::Vector{Float64}        # latitude dos clientes
    cy::Vector{Float64}        # longitude dos clientes
    distance::Matrix{Float64}  # matriz de distâncias entre todos os clientes
end

# estrutura com o resultado do problema
struct Reportproblem <: Data
    nome::String                                   # nome da heurística utilizada
    custo::Float64                                 # custo total da rota
    rota::AbstractVector{T} where {T <: Integer}   # rota gerada pela solução
end

# estrutura para os parametros do solver no modelo matemático do tsp
struct Solverparameters <: Data
    gap::Float64
    threads::Int64
    timelimit::Float64
end

#==========================================================================================#
# MÉTODOS DE APOIO                                                                         #
#==========================================================================================#

# esse método recebe a matriz binária da variável x e retorna a rota
function tour(mt::Matrix{Float64})
    rota::Vector{Int64} = []
    k = 1
    @inbounds for i = 1:size(mt,1)
        j = argmax(mt[k,:])
        push!(rota,j)
        k = j
    end
    return rota
end

# método que calcula a melhoria de na troca de pontos no algorítmo 2-opt
function prv(distmat::AbstractMatrix{T}, path::AbstractVector{S}, revLow::Int, revHigh::Int) where {T <: Real, S <: Integer}
    
    cost_delta = zero(eltype(distmat))
    
    # if there's an initial unreversed section
    if revLow > 1
        # new step onto the reversed section
        @inbounds cost_delta += distmat[path[revLow - 1], path[revHigh]]
        # no longer pay the cost of old step onto the reversed section
        @inbounds cost_delta -= distmat[path[revLow - 1], path[revLow]]
    end
    
    # The actual cost of walking along the reversed section doesn't change
    
    # because the distance matrix is symmetric.
    # if there's an unreversed section after the reversed bit
    if revHigh < length(path)
        # new step out of the reversed section
        @inbounds cost_delta += distmat[path[revLow], path[revHigh + 1]]
        # no longer pay the old cost of stepping out of the reversed section
        @inbounds cost_delta -= distmat[path[revHigh], path[revHigh + 1]]
    end
    return cost_delta
end

# método para destruir e reconstruir rotas
function routedestroy(data::Data)
    rota = [1:data.n...]
    rota = shuffle!(rota)
    custo = cost(data, rota)
    resultado = Reportproblem("", custo, rota)
    return resultado
end

# método que calcula a matriz de distâncias
@inbounds distance(cx::AbstractVector{T}, cy::AbstractVector{T}, n::Integer) where {T <: Real} = [i != j ? sqrt((cx[j] - cx[i])^2 + (cy[j] - cy[i])^2) : 1000.0 for i = 1:n, j = 1:n]

# método que calcula o custo da rota gerada
@inbounds cost(data::Dataproblem, route::AbstractVector{T}) where {T <: Integer} = sum([data.distance[route[i], route[i+1]] for i = 1:length(route)-1])

# método para geração de instâncias
@inbounds function generate_instance(n::Int)
    x = rand(-100.0:100.0, n) # latitude
    y = rand(-100.0:100.0, n) # longitude
    d = distance(x, y, n)     # matriz de distâncias
    return Dataproblem(n, x, y, d)
end

# método que calcula os savings das combinações das rotas
function savings(data::Dataproblem)
    c = copy(data.distance)
    s = zeros(Float64, data.n, data.n)
    n = copy(data.n)
    @inbounds for i = 1:n
        @inbounds for j = 2:i
            s[i, j] = c[i, 1] + c[1, j] - c[i,j]
            s[j, i] = c[j, 1] + c[1, i] - c[j,i]
        end
    end
    return s
end

#==========================================================================================#
# MÉTODOS DE SAÍDA                                                                         #
#==========================================================================================#

# método que retorna o gráfico da rota gerada
function graph(data::Dataproblem, resultado::Reportproblem, ativo::Bool)
    if ativo == true
        plt = []
        r = [resultado.rota[end]; resultado.rota[1:end]]
        scatter(data.cx, data.cy, size = (950, 800), legend = false, color = 2)
        scatter!((data.cx[r[1]], data.cy[r[1]]), markershape = :rect, color =:white)
        annotate!(data.cx, data.cy .+ 3, [1:data.n...], font(8, :white))
        title!("TSP : " * resultado.nome)
        xlabel!("Cost of route: " * string(round(Int, resultado.custo)))
        @inbounds for j = 1:length(r)-1
            plt = Plots.plot!( 
                ( 
                  [data.cx[r[j]], data.cx[r[j+1]]], 
                  [data.cy[r[j]], data.cy[r[j+1]]]  
                ), 
                arrow = true, 
                color = 1
            )
        end
        display(plt)
    end
end

# método que retorna o resultado da roteirização no terminal
function output(resultado::Reportproblem)
    rota = push!(resultado.rota, resultado.rota[1])
    println(" TSP PROBLEM ")
    println("==============: ")
    println("Algorithm     : ", resultado.nome)
    println("Cost of route : ", resultado.custo)
    println("Route         : ", rota)
    println("==============: ")
end
