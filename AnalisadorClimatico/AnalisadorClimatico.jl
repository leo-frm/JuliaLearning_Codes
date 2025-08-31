"""
# Analisador Climático
# Este módulo fornece funções para analisar dados climáticos, como calcular a média, encontrar a maior e menor temperatura, e gerar um relatório climático.
"""
module AnalisadorClimatico
using Plots
export gerar_grafico_climatico, ler_dados_de_arquivo, obter_dados_usuario, calcular_media_temperaturas, encontrar_maior_temperatura, encontrar_menor_temperatura, gerar_relatorio_climatico

"""
# analisador_climático.jl
"""

"""
    gerar_grafico_climatico(dados_temperaturas, caminho_arquivo_saida="grafico_climatico.png")
Gera um gráfico das temperaturas fornecidas e salva em um arquivo.
    argumentos:
        - dados_temperaturas: um vetor de números representando as temperaturas.
        - caminho_arquivo_saida: o caminho do arquivo onde o gráfico será salvo (padrão é "grafico_climatico.png").
    retorna:
        - nada, mas salva o gráfico em um arquivo. 
"""

    gerar_grafico_climatico(dados_temperaturas, caminho_arquivo_saida::String="grafico_climatico.png") = begin
    if isempty(dados_temperaturas)
        println("Nenhum dado de temperatura fornecido para gerar o gráfico.")
        return
    end
        # Configura o backend do Plots
        gr()  # Usando o backend GR para gráficos
        dias = 1:length(dados_temperaturas)  # Cria um vetor de dias baseado no número de temperaturas
        plot(dias, dados_temperaturas,
            label="Temperatura Diária",
            title="Temperaturas Registradas",
            xlabel="Dia",
            ylabel="Temperatura (°C)",
            legend=:topleft,
            lw=2,  # Espessura da linha
            marker=:circle  # Adiciona um marcador em cada ponto de dados
        )
        try
            savefig(caminho_arquivo_saida)
            println("Gráfico salvo como '$(caminho_arquivo_saida)'!")
        catch e
            println("Erro ao salvar o gráfico: ", e)
        end
    end

"""
    ler_dados_de_arquivo(caminho_arquivo::String)
Lê um arquivo de texto contendo temperaturas, uma por linha, e retorna um vetor de números.
    argumentos:
        - caminho_arquivo: o caminho do arquivo a ser lido.
    retorna:
        - um vetor de números representando as temperaturas lidas do arquivo.
"""

function ler_dados_de_arquivo(caminho_arquivo::String)
    temperaturas = Float64[]
    try
        open(caminho_arquivo, "r") do file_stream
        for linha_atual in eachline(file_stream)
            try
                temp = parse(Float64, linha_atual)
                push!(temperaturas, temp)
            catch e
                println("Atenção: a linha '", linha_atual, "' não é um número válido.")
            end
        end
    end
    catch e
        println("Erro ao ler o arquivo: ", e)
        return Float64[]  # Retorna um vetor vazio em caso de erro
    end
    return temperaturas
end

"""
    obter_dados_usuario()
Solicita ao usuário que insira temperaturas separadas por espaço e retorna um vetor de números.
    argumentos:
        - nenhum
    retorna:
        - um vetor de números representando as temperaturas inseridas pelo usuário.
"""


function obter_dados_usuario()
    temperaturas_nmr = Float64[]
    println("Por favor, insira as temperaturas separadas por espaço:")
    entrada = readline()
    temperaturas_texto = split(entrada)
    for temp in temperaturas_texto
        try
            temp_nmr = parse(Float64, temp)
            push!(temperaturas_nmr, temp_nmr)
        catch e
            println("Atenção: o valor '", temp, "' não é um número válido.")
        end
    end
    return temperaturas_nmr
end

"""
   calcular_media_temperaturas(dados_temperaturas)
Calcula a média das temperaturas fornecidas.
    argumentos:
        - dados_temperaturas: um vetor de números representando as temperaturas.
    retorna:   
        - a média das temperaturas, ou `nothing` se o vetor estiver vazio.
"""

function calcular_media_temperaturas(dados_temperaturas)
    if isempty(dados_temperaturas)
        return nothing
    end
    soma = sum(dados_temperaturas)
    nmr_dados = length(dados_temperaturas)
    media_temperaturas = soma/nmr_dados
    return media_temperaturas
end    

"""
    encontrar_maior_temperatura(dados_temperaturas)
Encontra a maior temperatura entre os dados fornecidos.
    argumentos:
        - dados_temperaturas: um vetor de números representando as temperaturas.
    retorna:    
        - a maior temperatura, ou `nothing` se o vetor estiver vazio.
"""

function encontrar_maior_temperatura(dados_temperaturas)
    if isempty(dados_temperaturas)
        return nothing
    end
    maior = maximum(dados_temperaturas)
    return maior
end

"""
    encontrar_menor_temperatura(dados_temperaturas)
Encontra a menor temperatura entre os dados fornecidos.
    argumentos:
        - dados_temperaturas: um vetor de números representando as temperaturas.
    retorna:    
        - a menor temperatura, ou `nothing` se o vetor estiver vazio.
"""

function encontrar_menor_temperatura(dados_temperaturas)
    if isempty(dados_temperaturas)
        return nothing
    end
    menor = minimum(dados_temperaturas)
    return menor
end

"""
    gerar_relatorio_climatico(dados_temperaturas)
Gera um relatório climático com as temperaturas fornecidas, incluindo a média, a maior e a menor temperatura.
    argumentos:
        - dados_temperaturas: um vetor de números representando as temperaturas.
        - caminho_arquivo_saida: um caminho de arquivo onde o relatório será salvo (opcional).
    retorna:    
        - nada, mas imprime o relatório no console ou salva em um arquivo se o caminho for fornecido.
"""

function gerar_relatorio_climatico(dados_temperaturas, caminho_arquivo_saida::Union{String, Nothing}=nothing)
    if isempty(dados_temperaturas)
        println("Nenhum dado de temperatura fornecido.")
        return
    end
    
    media = calcular_media_temperaturas(dados_temperaturas)
    maior = encontrar_maior_temperatura(dados_temperaturas)
    menor = encontrar_menor_temperatura(dados_temperaturas)

    linhas_relatório = [
        "Temperaturas registradas: $(dados_temperaturas)",
        "Média das temperaturas: $(media) °C",
        "Maior temperatura: $(maior) °C",
        "Menor temperatura: $(menor) °C"
    ]

    if caminho_arquivo_saida !== nothing
        try
        open(caminho_arquivo_saida, "w") do file_stream
            for linha in linhas_relatório
                println(file_stream, linha)
            end
        end
        catch e
            println("Erro ao salvar o relatório em arquivo: ", e)
            return
        end
        println("Dados salvos em ", caminho_arquivo_saida)
    else
        println("Relatório não salvo em arquivo.")
        for linha in linhas_relatório
            println(linha)
        end
    end
end

end  # module AnalisadorClimatico