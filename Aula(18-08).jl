using LinearAlgebra, Random

parametro = (Δ = 1.0,
            Δ_min = 1e-4,
            K_max = 5000,
            M = 10,
            N = 20,
            λ = 0.3)

function f_minimo(A, b, x, λ)
    return norm(A * x - b)^2 + λ * norm(x, 1)
end

A = randn(parametro.M, parametro.N)

x_verdadeiro = zeros(parametro.N)
posicoes_nao_nulas = [5, 12, 18]  # Posições escolhidas
valores_nao_nulos = [2.5, -1.8, 3.2]  # Valores escolhidos
x_verdadeiro[posicoes_nao_nulas] = valores_nao_nulos
b = A * x_verdadeiro

x = zeros(parametro.N)
f0 = f_minimo(A, b, x, parametro.λ)
Δ = parametro.Δ
k = 1

while k <= parametro.K_max && Δ >= parametro.Δ_min
    global x, f0, Δ, k  
    x_alterado = false

    for i in 1:parametro.N
        ei = zeros(parametro.N)
        ei[i] = 1.0
        f_i = f_minimo(A, b, x + Δ * ei, parametro.λ)
        if f_i < f0
            x = x + Δ * ei
            f0 = f_i
            x_alterado = true
        else
            f_i_neg = f_minimo(A, b, x - Δ * ei, parametro.λ)
            if f_i_neg < f0
                x = x - Δ * ei
                f0 = f_i_neg
                x_alterado = true
            end
        end
    end
    if !x_alterado
        Δ = Δ / 2
    end
    k += 1 
end

# Resultados
println("Iterações: ", k-1)
println("Erro final: ", norm(x - x_verdadeiro))
println("x_verdadeiro: ", x_verdadeiro[posicoes_nao_nulas])
println("x_estimado:   ", x[posicoes_nao_nulas])