using LinearAlgebra, Random

parametro = (Δ = 1.0,
            Δ_min = 1e-4,
            K_max = 5000,
            M = 10,
            N = 20,
            λ = 0.3,
            threshold = 1e-3)  # Limiar para thresholding

function f_minimo(A, b, x, λ)
    return norm(A * x - b)^2 + λ * norm(x, 1)
end

function soft_thresholding(x, threshold)
    """Aplica soft thresholding: zera elementos menores que o limiar"""
    return sign.(x) .* max.(abs.(x) .- threshold, 0)
end

function hard_thresholding(x, threshold)
    """Aplica hard thresholding: zera elementos com valor absoluto menor que o limiar"""
    x_thresh = copy(x)
    x_thresh[abs.(x_thresh) .< threshold] .= 0
    return x_thresh
end

# Configuração do problema
Random.seed!(42)  # Para reprodutibilidade
A = randn(parametro.M, parametro.N)
x_verdadeiro = zeros(parametro.N)
posicoes_nao_nulas = [5, 12, 18]
valores_nao_nulos = [2.5, -1.8, 3.2]
x_verdadeiro[posicoes_nao_nulas] = valores_nao_nulos
b = A * x_verdadeiro

# Algoritmo original
println("=== ALGORITMO ORIGINAL ===")
x_original = zeros(parametro.N)
f0 = f_minimo(A, b, x_original, parametro.λ)
Δ = parametro.Δ
k = 1

while k <= parametro.K_max && Δ >= parametro.Δ_min
    global x_original, f0, Δ, k
    x_alterado = false
    
    for i in 1:parametro.N
        ei = zeros(parametro.N)
        ei[i] = 1.0
        
        # Testa direção positiva
        f_i = f_minimo(A, b, x_original + Δ * ei, parametro.λ)
        if f_i < f0
            x_original = x_original + Δ * ei
            f0 = f_i
            x_alterado = true
        else
            # Testa direção negativa
            f_i_neg = f_minimo(A, b, x_original - Δ * ei, parametro.λ)
            if f_i_neg < f0
                x_original = x_original - Δ * ei
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

println("Iterações: ", k-1)
println("Erro final: ", norm(x_original - x_verdadeiro))
println("x_verdadeiro: ", x_verdadeiro[posicoes_nao_nulas])
println("x_estimado:   ", x_original[posicoes_nao_nulas])

# Algoritmo com hard thresholding
println("\n=== ALGORITMO COM HARD THRESHOLDING ===")
x_thresh = zeros(parametro.N)
f0 = f_minimo(A, b, x_thresh, parametro.λ)
Δ = parametro.Δ
k = 1

while k <= parametro.K_max && Δ >= parametro.Δ_min
    global x_thresh, f0, Δ, k
    x_alterado = false
    
    for i in 1:parametro.N
        ei = zeros(parametro.N)
        ei[i] = 1.0
        
        # Testa direção positiva
        x_temp = x_thresh + Δ * ei
        x_temp = hard_thresholding(x_temp, parametro.threshold)
        f_i = f_minimo(A, b, x_temp, parametro.λ)
        
        if f_i < f0
            x_thresh = x_temp
            f0 = f_i
            x_alterado = true
        else
            # Testa direção negativa
            x_temp = x_thresh - Δ * ei
            x_temp = hard_thresholding(x_temp, parametro.threshold)
            f_i_neg = f_minimo(A, b, x_temp, parametro.λ)
            
            if f_i_neg < f0
                x_thresh = x_temp
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

println("Iterações: ", k-1)
println("Erro final: ", norm(x_thresh - x_verdadeiro))
println("x_verdadeiro: ", x_verdadeiro[posicoes_nao_nulas])
println("x_estimado:   ", x_thresh[posicoes_nao_nulas])

# Algoritmo com thresholding apenas no final
println("\n=== ALGORITMO COM THRESHOLDING FINAL ===")
x_final = hard_thresholding(x_original, parametro.threshold)
println("Erro final com thresholding: ", norm(x_final - x_verdadeiro))
println("x_verdadeiro: ", x_verdadeiro[posicoes_nao_nulas])
println("x_estimado:   ", x_final[posicoes_nao_nulas])

# Comparação dos resultados
println("\n=== COMPARAÇÃO ===")
println("Erro original: ", norm(x_original - x_verdadeiro))
println("Erro com hard thresholding durante: ", norm(x_thresh - x_verdadeiro))
println("Erro com thresholding final: ", norm(x_final - x_verdadeiro))

# Análise da esparsidade
println("\n=== ANÁLISE DA ESPARSIDADE ===")
println("Elementos não-nulos no x_verdadeiro: ", sum(abs.(x_verdadeiro) .> 1e-10))
println("Elementos não-nulos no x_original: ", sum(abs.(x_original) .> 1e-3))
println("Elementos não-nulos no x_thresh: ", sum(abs.(x_thresh) .> 1e-10))
println("Elementos não-nulos no x_final: ", sum(abs.(x_final) .> 1e-10))