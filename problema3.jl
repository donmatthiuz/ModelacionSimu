using JuMP
using HiGHS

# Matriz de costos (usamos 9999 como valor prohibido)
INF = 9999
cost = [
    50   50    INF  20;
    70   40    20   30;
    90   30    50   INF;
    70   20    60   70
]

n_trabajadores, n_puestos = size(cost)

model = Model(HiGHS.Optimizer)

@variable(model, x[1:n_trabajadores, 1:n_puestos], Bin)

# Función objetivo: minimizar el costo total
@objective(model, Min, sum(cost[i,j] * x[i,j] for i in 1:n_trabajadores, j in 1:n_puestos))

# Restricción: cada trabajador solo en un puesto
@constraint(model, [i=1:n_trabajadores], sum(x[i,j] for j in 1:n_puestos) == 1)

# Restricción: cada puesto solo asignado a un trabajador
@constraint(model, [j=1:n_puestos], sum(x[i,j] for i in 1:n_trabajadores) == 1)

# Eliminar asignaciones prohibidas (INF)
for i in 1:n_trabajadores, j in 1:n_puestos
    if cost[i,j] == INF
        fix(x[i,j], 0.0; force=true)
    end
end

optimize!(model)

println("Resultado:")
for i in 1:n_trabajadores, j in 1:n_puestos
    if value(x[i,j]) > 0.5
        println("  Trabajador $i → Puesto $j (Costo: \$$(cost[i,j]))")
    end
end

println("Costo total mínimo: \$", objective_value(model))
