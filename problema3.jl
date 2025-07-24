using JuMP, HiGHS, Printf

# a) Formular el problema de programamcion lineal.

# Sabemos que tenemos 6 horarios y el $$x_i$$ representa los buses entonces la funcion a minimizar sera la suma de los buses


#$$F(z) = x_1 + x_2 + x_3 + x_4 + x_5 + x_6$$ 

# Esto sera la funcion objetivo a minimizar ya que es la suma de los buses y las restricciones son los horarios

# 12:00-4:00 AM: x₁ + x₆ ≥ 4
# 4:00-8:00 AM: x₁ + x₂ ≥ 8 
# 8:00 AM-12:00 PM: x₂ + x₃ ≥ 10
# 12:00-4:00 PM: x₃ + x₄ ≥ 7 
# 4:00-8:00 PM: x₄ + x₅ ≥ 12 
# 8:00 PM-12:00 AM: x₅ + x₆ ≥ 4

# Esto conforme a lo que se presento en la grafica ya que la suma de los buses de los turnos tiene que cumplir minimo con el requisito de la demanda, y la demanda es el numero despues del ≥
# La restriccion de no negatividad es que siempre existan buses

# Ahora solo lo convertimos a un problema de programacon con lua

model = Model()


@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@variable(model, x3 >= 0)
@variable(model, x4 >= 0)
@variable(model, x5 >= 0)
@variable(model, x6 >= 0)

@constraint(model, x1 + x6 >= 4)
@constraint(model, x1 + x2 >= 8)
@constraint(model, x2 + x3 >= 10)
@constraint(model, x3 + x4 >= 7)
@constraint(model, x4 + x5 >= 12)
@constraint(model, x5 + x6 >= 4)

@objective(model, Min, x1 + x2 + x3 + x4 + x5 + x6 )

set_optimizer(model, HiGHS.Optimizer)


optimize!(model)




print("x1 = ", value.(x1), "\n")
print("x2 = ", value.(x2), "\n")
print("x3 = ", value.(x3), "\n")
print("x4 = ", value.(x4), "\n\n")
print("x5 = ", value.(x5), "\n\n")
print("x6 = ", value.(x6), "\n\n")

print("z = ", objective_value(model))


# Por lo que el total minimo de buses para que todo se cumpla con 26