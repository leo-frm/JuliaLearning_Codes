using DataFrames, Dates, Plots, DelimitedFiles, Statistics, CSV

struct RegistroMeteorologico
    data::Date
    cidade::String
    temperatura::Float64
    umidade::Float64
    velocidade_vento::Float64 #km/h
    pressao::Float64 #hPa
end

function ler_registros(caminho_arquivo::String)
    open(caminho_arquivo, "r") do file_stream
        registros = RegistroMeteorologico[]
        for linha in eachline(file_stream)
            try
                partes = split(linha, ",")
                data = Date(partes[1], "yyyy-mm-dd")
                cidade = partes[2]
                temperatura = parse(Float64, partes[3])
                umidade = parse(Float64, partes[4])
                velocidade_vento = parse(Float64, partes[5])
                pressao = parse(Float64, partes[6])
                push!(registros, RegistroMeteorologico(data, cidade, temperatura, umidade, velocidade_vento, pressao))
            catch e
                println("Erro ao processar a linha: ", linha)
            end
        end
        return registros
    end
end

function estatistica_por_cidade(registros::Vector{RegistroMeteorologico}, campo::Symbol, cidade::String)
    valores = [getfield(r, campo) for r in registros if r.cidade == cidade]
    if isempty(valores)
        return (max=NaN, min=NaN, med=NaN, var=NaN)
    else
        class_max = maximum(valores)
        class_min = minimum(valores)
        class_med = mean(valores)
        variacao = class_max - class_min
    end
    return (max=class_max, min=class_min, med=class_med, var=variacao)
end

function relatorio_meteorologico_por_cidade(registros::Vector{RegistroMeteorologico}, cidade::String)
    temp_info = estatistica_por_cidade(registros, :temperatura, cidade)
    umid_info = estatistica_por_cidade(registros, :umidade, cidade)
    vel_info = estatistica_por_cidade(registros, :velocidade_vento, cidade)
    pres_info = estatistica_por_cidade(registros, :pressao, cidade)

    relatorio = """
                Relatório Meteorológico para a cidade de $cidade:
                Temperaturas:
                - Máxima: $(temp_info.max) °C
                - Mínima: $(temp_info.min) °C
                - Média: $(temp_info.med) °C
                - Variação: $(temp_info.var) °C
                Umidade:
                - Máxima: $(umid_info.max) %
                - Mínima: $(umid_info.min) %
                - Média: $(umid_info.med) %
                - Variação: $(umid_info.var) %
                Velocidade do Vento:
                - Máxima: $(vel_info.max) km/h
                - Mínima: $(vel_info.min) km/h
                - Média: $(vel_info.med) km/h
                - Variação: $(vel_info.var) km/h
                Pressão Atmosférica:
                - Máxima: $(pres_info.max) hPa
                - Mínima: $(pres_info.min) hPa
                - Média: $(pres_info.med) hPa
                - Variação: $(pres_info.var) hPa
                """
    return relatorio
end

function encontrar_cidade_max(registros::Vector{RegistroMeteorologico}, campo::Symbol, criterio::Symbol)
    cidades = unique([r.cidade for r in registros])
    if isempty(cidades)
        return "Nenhuma cidade encontrada nos registros."
    end
    resultados = [(cidade=c, valor = getfield(estatistica_por_cidade(registros, campo, c), criterio)) for c in cidades]
    resultados_validados = filter(r -> !isnan(r.valor), resultados)
    if isempty(resultados_validados)
        return "Nenhuma cidade com dados válidos de temperatura."
    end
    valores_numericos = [r.valor for r in resultados_validados]
    valores_numericos, idx = findmax(valores_numericos)
    cidade_max = resultados_validados[idx]
    return "A cidade com a maior $(campo) é $(cidade_max.cidade) com um valor de $(cidade_max.valor)."
end

function plot_graficos_meteorologicos(registros::Vector{RegistroMeteorologico}, cidade::String)
    gr()
    cidade_registros = filter(r -> r.cidade == cidade, registros)
    if isempty(cidade_registros)
        println("Nenhum dado encontrado para a cidade: $cidade")
        return
    end

    datas = [r.data for r in cidade_registros]
    temperaturas = [r.temperatura for r in cidade_registros]
    umidades = [r.umidade for r in cidade_registros]
    velocidades_vento = [r.velocidade_vento for r in cidade_registros]
    pressoes = [r.pressao for r in cidade_registros]

    p_temp = plot(datas, temperaturas, ylabel = "°C", title="Temperatura", lw=2, marker=:circle)
    p_umid = plot(datas, umidades, ylabel = "%", title="Umidade", lw=2, marker=:circle)
    p_vel_vent = plot(datas, velocidades_vento, ylabel = "km/h", title="Velocidade do Vento", lw=2, marker=:circle)
    p_pressao = plot(datas, pressoes, ylabel = "hPa", title="Pressão Atmosférica", lw=2, marker=:circle)

    plot(p_temp, p_umid, p_vel_vent, p_pressao, legend=false, layout = (2, 2), size = (1200, 800), plot_title = "Dados Meteorológicos de $cidade")
end

function imprimir_relatorio(relatorio::String)
    open("relatorio_meteorologico.txt", "a") do file
        write(file, relatorio * "\n")
    end
end

function imprimir_grafico(cidade::String, dados::Vector{RegistroMeteorologico})
    plot_graficos_meteorologicos(dados, cidade)
    savefig("graficos_meteorologicos_$cidade.png")
    println("Gráfico meteorológico para $cidade salvo como 'graficos_meteorologicos_$cidade.png'.")
end

dados = ler_registros("dados.csv")

imprimir_relatorio(relatorio_meteorologico_por_cidade(dados, "São Paulo"))
imprimir_relatorio(relatorio_meteorologico_por_cidade(dados, "Rio de Janeiro"))
imprimir_relatorio(relatorio_meteorologico_por_cidade(dados, "Belo Horizonte"))
imprimir_grafico("São Paulo", dados)
imprimir_grafico("Rio de Janeiro", dados)
imprimir_grafico("Belo Horizonte", dados)

println(encontrar_cidade_max(dados, :temperatura, :max))
println(encontrar_cidade_max(dados, :velocidade_vento, :med))
