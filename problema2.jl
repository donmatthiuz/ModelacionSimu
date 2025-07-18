using JuMP, HiGHS, Printf

# Datos del problema
demanda     = [180, 250, 190, 140, 220, 250]
costo_prod  = [50,  45,  55,  52,  48,  50]
costo_almac = [ 8,  10,  10,  10,   8,   8]
capacidad   = 225
n           = length(demanda)

# Modelo
model = Model(HiGHS.Optimizer)
@variable(model, x[1:n] >= 0)   # producción mes t
@variable(model, I[1:n] >= 0)   # inventario al cierre mes t

# Inventario inicial y balance
@constraint(model, I[1] == x[1] - demanda[1])
for t in 2:n
    @constraint(model, I[t] == I[t-1] + x[t] - demanda[t])
end

# Límite de producción
for t in 1:n
    @constraint(model, x[t] <= capacidad)
end

# Función objetivo
@objective(model, Min,
    sum(costo_prod[t]*x[t] + costo_almac[t]*I[t] for t in 1:n)
)

optimize!(model)

xp = value.(x)
Ip = value.(I)

println(" Mes │ Producción │ Inventario")
println("─────┼────────────┼────────────")
for t in 1:n
    @printf(" %2d  │ %10.1f │ %10.1f\n", t, xp[t], Ip[t])
end

costo_opt   = sum(costo_prod[t]*xp[t] + costo_almac[t]*Ip[t] for t in 1:n)
costo_noopt = sum(costo_prod[t]*demanda[t] for t in 1:n)
ahorro      = costo_noopt - costo_opt

println("\nCosto [no optimo]: \$", round(costo_noopt, digits=2))
println("Costo [optimo]    : \$", round(costo_opt,  digits=2))
println("Ahorro obtenido       : \$", round(ahorro,     digits=2))

