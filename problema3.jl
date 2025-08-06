using JuMP
using HiGHS

# Matriz de costos (usamos 9999 como valor prohibido)
INF = 9999
cost = [
    3  8  2 10  3  9;
    2  2  7  6  5  7;
    5  6  4  5  6  6;
    4  2  7  5  4  7;
    10 3  4  2  3  5;
    3  5  4  2  3  8
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
