using JuMP, CPLEX

# Número de cidades
n = 5

# Distâncias entre as cidades
d = [
    0 10 15 20 25;
    10 0 35 25 30;
    15 35 0 30 35;
    20 25 30 0 10;
    25 30 35 10 0
]

# Criar o modelo de otimização
m = Model(CPLEX.Optimizer)

# Criar as variáveis de decisão
@variable(m, x[1:n, 1:n], Bin)

# Definir a função objetivo como a soma das distâncias entre as cidades
@objective(m, Min, sum(d[i,j] * x[i,j] for i in 1:n, j in 1:n))

# Adicionar a restrição de fluxo de entrada/saída
@constraint(m, [j = 1:n], sum(x[i,j] for i = 1:n if i != j) == 1)
@constraint(m, [i = 1:n], sum(x[i,j] for j = 1:n if i != j) == 1)

# Definir a função de corte para a restrição de sub-rotas
function mycutcallback(cb_data)
    for i in 1:n, j in 1:n, k in 1:n
        if i != j && i != k && j != k
            if callback_value(cb_data, x[i,j]) + callback_value(cb_data, x[j,k]) + callback_value(cb_data, x[k,i]) > 2
                println(callback_value(cb_data, x[i,j]))
                println(callback_value(cb_data, x[j,k]))
                cut = @build_constraint(x[i,j] + x[j,k] + x[k,i] <= 2) # Adicionar um corte/restrição para garantir que não haja subrotas
                MOI.submit(m, MOI.UserCut(cb_data), cut)               # adiciona a nova restrição/corte ao modelo
            end
        end
    end
    return
end

# Adicionar a função de corte ao modelo de otimização
MOI.set(m, MOI.UserCutCallback(), mycutcallback)

# Resolver o problema
optimize!(m)

# solução ótima é 95.0
objective_value(m)

value.(x)






















##################################33

# Criar o modelo
model = Model(GLPK.Optimizer)

# # Adicionar as variáveis x[i,j]
@variable(model, x[1:n, 1:n], Bin)

# # Adicionar a restrição de sub-ciclos
for i in 1:n
    @constraint(model, sum(x[i,j] for j in 1:n if i != j) == 1)
    @constraint(model, sum(x[j,i] for j in 1:n if i != j) == 1)
end

# # Adicionar a restrição de sub-rotas
for i in 1:n, j in 1:n, k in 1:n
    if i != j && i != k && j != k
        @constraint(model, x[i,j] + x[j,k] + x[k,i] <= 2)
    end
end

# # Adicionar a função objetivo
@objective(model, Min, sum(d[i,j]*x[i,j] for i in 1:n, j in 1:n))

# # Resolver o modelo
optimize!(model)

# # Imprimir a solução
# println("Custo total da rota: ", objective_value(model))
for i in 1:n, j in 1:n
    if value(x[i,j]) > 0
        println("Vá de cidade ", i, " para cidade ", j)
    end
end

value.(x)