"""
    # Main entry point for the Julia package
"""

include("AnalisadorClimatico.jl")
using .AnalisadorClimatico

function main()
    println("Bem-vindo ao Analisador Climático!")

    temps = ler_dados_de_arquivo("temperaturas.txt")
    if isempty(temps)
        println("Não foi possível ler dados de temperatura do arquivo, solicitando entrada do usuário.")
        temps = obter_dados_usuario()
    else
        gerar_relatorio_climatico(temps, "relatorio_climatico.txt")
        gerar_grafico_climatico(temps, "grafico_climatico.png")
    end
    println("Obrigado por usar o Analisador Climático!")
end

main()