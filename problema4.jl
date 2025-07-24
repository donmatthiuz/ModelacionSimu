using JuMP, HiGHS, Printf

# a) Formular el problema de programamcion lineal.

# Para esto debemos de ver las variables son 

#x1 casas sencillas, x2 casas dobles x3 casas triples y x4 casas cuadruples
#Luego la funcion objetivo es maximizar UTILIDAD = INGRESOS - COSTOS
# Y entonces los INGRESOS = Impuestos recaudados por todas las unidades
# COSTOS = Costos de construcción + Costos de demolición

# Ahora vamos a colocar la funcion

# Ingresos = 1000x1 + 1900x2 + 2700x3 + 3400x4

# Funcion a maximizar
# INGRESOS = (1000x1 + 1900x2 + 2700x3 + 3400x4)

# Ahora las restricciones

# La primera restriccion es de la financiacion no exceda los 15M
# Entonces 50000x1 + 70000x2 + 130000x3 + 160000x4 <= 15000000 - 600000
# ESTOS 600K son de lo que ya se gastaron en demolres las 300 casas a 2000 c/u osea 300*2000 = 600000

# La segunda es el espacio, para ello calculamos el espacio total
# 300 * 0.25 acres , lo cual nos da 75 acres de total a eso le restamos el 15% de espacios para cosas que no son casas
# esto nos da 63.75 acres

# Ahora que tenemos el espacio completo definimos el espacio de cada casas

#0.18x1 + 0.28x2 + 0.4x3 +0.5x4 <= 63.75  que es la restriccion del espacio
# solo nos falta la restriccion de cada tipo de casas

# unifamiliar tiene que >= 20% osea
# 0.18x1 >= 0.2*63.75

# las unidades triples y cuadruples ocupan por lo menos 25% osea
# 0.4x3 + 0.5x4 >= 0.25 * 63.75

# Las unidades dobles deben ocupar un minimo de 10%. osea
# 0.28x2 >= 0.1*63.75




# b) Resolver el problema usando la librerıa JuMP o Pulp, en variables continuas, y determinar la distribucion optima.



model_continuas = Model()

@variable(model_continuas, x1 >= 0)
@variable(model_continuas, x2 >= 0)
@variable(model_continuas, x3 >= 0)
@variable(model_continuas, x4 >= 0)



# Financiamiento
@constraint(model_continuas, 50000x1 + 70000x2 + 130000x3 + 160000x4 <= 14400000)
# espacio
@constraint(model_continuas, 0.18x1 + 0.28x2 + 0.4x3 +0.5x4 <= 63.75)

#espacio unifarmiliares
# 0.18x1 >= 0.2*63.75
@constraint(model_continuas, 0.18x1 >= 12.75)

# 0.4x3 + 0.5x4 >= 0.25 * 63.75
@constraint(model_continuas, 0.4x3 + 0.5x4 >= 15.9375)

# 0.28x2 >= 0.1*63.75
@constraint(model_continuas, 0.28x2 >= 6.375)




@objective(model_continuas, Max, 1000x1 + 1900x2 + 2700x3 + 3400x4)



set_optimizer(model_continuas, HiGHS.Optimizer)


optimize!(model_continuas)

print("x1 = ", value.(x1), "\n")
print("x2 = ", value.(x2), "\n")
print("x3 = ", value.(x3), "\n")
print("x4 = ", value.(x4), "\n\n")


print("z = ", objective_value(model_continuas))


# Se recaudra 335505.95 dolares si se construye 70 casas unifamiliares , 82 dobles, 0 triples y 31 cuadruples

#  c) Resolver el problema en variables enteras, y comparar la solucion con la de (b). Discuta sus resultados.


model_enteras = Model()

@variable(model_enteras, x1_e >= 0, Int)
@variable(model_enteras, x2_e >= 0, Int)
@variable(model_enteras, x3_e >= 0, Int)
@variable(model_enteras, x4_e >= 0, Int)

# Financiamiento
@constraint(model_enteras, 50000x1_e + 70000x2_e + 130000x3_e + 160000x4_e <= 14400000)

# Espacio total
@constraint(model_enteras, 0.18x1_e + 0.28x2_e + 0.4x3_e + 0.5x4_e <= 63.75)

# Espacio unifamiliares
@constraint(model_enteras, 0.18x1_e >= 0.2 * 63.75)
@constraint(model_enteras, 0.4x3_e + 0.5x4_e >= 0.25 * 63.75)
@constraint(model_enteras, 0.28x2_e >= 0.1 * 63.75)

# Función objetivo
@objective(model_enteras, Max, 1000x1_e + 1900x2_e + 2700x3_e + 3400x4_e)

set_optimizer(model_enteras, HiGHS.Optimizer)

optimize!(model_enteras)

# Resultados
println("x1_e = ", value(x1_e))
println("x2_e = ", value(x2_e))
println("x3_e = ", value(x3_e))
println("x4_e = ", value(x4_e))
println("\nz = ", objective_value(model_enteras))



# Se recaudra  334700.0 dolares si se construye 72 casas unifamiliares , 81 dobles, 0 triples y 32 cuadruples


# Discusion

# En el modelo entero al redondear variables causo que se pierda espacio o presupuesto que no puede usarse en una casa completa.
# Pero en escencia ambos concluyeron que no conviene hacer ninguna casa triple.


