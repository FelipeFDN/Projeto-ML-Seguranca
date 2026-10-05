# Aplicação Streamlit e atualização meteorológica

## O que foi integrado

- `app/streamlit_app.py` apresenta somente a malha vetorial dos 295 municípios de Santa Catarina como objetos SVG próprios, sem biblioteca cartográfica, tiles ou mapa externo. As longitudes são corrigidas pelo cosseno da latitude média do estado para preservar proporções geográficas. O limite oficial de Florianópolis é um `MultiPolygon` com a ilha principal e outras ilhas menores; seu formato alongado representa o município inteiro, não apenas a área urbana. O nome aparece ao passar o mouse como texto sem caixa; clicar no objeto municipal seleciona a cidade e enquadra seus bounds completos. A cidade selecionada recebe preenchimento coral vivo e opaco, com contorno mais escuro; clicar novamente limpa a seleção e retorna ao enquadramento estadual. A geometria é cacheada e enviada uma vez por carga do componente; interações seguintes enviam apenas cidade e evento, mantendo os mesmos paths no iframe.
- O cartão de maior estimativa mostra o desastre líder somente quando sua saída ultrapassa 70%; em 70% ou menos, apresenta “Baixa chance de desastres”.
- Após a seleção, a interface consulta as últimas 24 horas disponíveis no Open-Meteo, agrega precipitação total/máxima, temperatura média, umidade média, vento máximo e pressão média e envia esses campos junto com a estação INMET associada à cidade para o modelo.
- `fetch_recent_weather` usa cache Streamlit de 43.200 segundos (12 horas). A malha e as coordenadas das sedes são cacheadas em disco; a consulta meteorológica ocorre sob demanda ao selecionar uma cidade, em vez de baixar dados para todos os municípios a cada atualização.
- O notebook `src/main.ipynb` exporta `models/best_model.joblib` depois da avaliação. A escolha usa o maior `f1_macro`; o candidato vencedor é retreinado com todo o conjunto de treino antes de ser salvo. O artefato guarda o estimador, colunas, métricas, data de treino e relação município-estação.

## Executar

Na raiz do projeto e com o ambiente virtual ativo:

```powershell
pip install -r requirements.txt
```

Execute todas as células de `src/main.ipynb` para avaliar e exportar o modelo. Depois:

```powershell
streamlit run app/streamlit_app.py
```

O serviço requer acesso à internet para baixar a malha do IBGE e consultar o Open-Meteo. A interface avisa quando o arquivo do modelo ainda não foi criado.

## Fonte meteorológica e atualização

### Fonte usada pelo app: Open-Meteo

O app consulta o endpoint Forecast por latitude/longitude, com `past_days=1`, `forecast_days=1` e as variáveis horárias precipitação, temperatura, umidade, vento e pressão. A janela das últimas 24 horas é agregada para as features do modelo: precipitação total e máxima, temperatura e umidade médias, vento máximo e pressão média. O vento é solicitado em m/s.

`fetch_recent_weather` usa cache por coordenada durante 43.200 segundos (12 horas). A consulta é sob demanda, somente para o município selecionado. Os valores passados do Open-Meteo vêm de modelos meteorológicos e não são observações oficiais do INMET; a janela de 24 horas aproxima a agregação do treino, mas fonte, ponto, resolução e definições das variáveis podem divergir. Reiniciar o processo Streamlit limpa o cache.

### Fonte oficial INMET: catálogo e observações

O catálogo público `https://apitempo.inmet.gov.br/estacoes/T` retorna metadados das estações, incluindo código (`CD_ESTACAO`), nome, coordenadas, UF, tipo e situação operacional. Assim, a escolha da estação mais próxima é viável sem credenciais: filtrar `SG_ESTADO == "SC"`, `TP_ESTACAO == "Automatica"` e `CD_SITUACAO == "Operante"`, calcular a distância geodésica entre a sede municipal e cada estação e selecionar a menor. A situação deve ser reconsultada, pois pode mudar. Na consulta feita em 04/10/2026, o catálogo listava 672 estações; em Florianópolis, a A806 estava como `Pane` e Rancho Queimado (A870), a cerca de 49,5 km, era a automática `Operante` mais próxima. Esses números são um retrato daquele momento, não uma configuração fixa.

O site oficial da [Tabela de Dados das Estações](https://tempo.inmet.gov.br/TabelaEstacoes) obtém as observações recentes pelo POST `https://apitempo.inmet.gov.br/estacao/front/`, enviando intervalo de datas, código da estação, `seed` e `gcap`. O `gcap` é um token Google reCAPTCHA gerado pelo frontend oficial. Uma chamada direta de servidor sem esse token retornou HTTP 200 com corpo de erro (`{"error":"Algo deu errado!!"}`); portanto o catálogo é público, mas a leitura recente não é uma API anônima adequada para o backend do Streamlit. Não se deve automatizar nem contornar o reCAPTCHA. Os arquivos públicos de dados históricos do INMET continuam disponíveis, mas não equivalem a uma API de observações em tempo real.

O app ainda não consome as observações recentes do INMET. Para trocar a fonte, é necessário obter do INMET uma forma autorizada de acesso automatizado; depois, implementar um adaptador que consulte o catálogo, escolha a estação `Operante` mais próxima, converta as variáveis/unidades oficiais para as features do treino, trate lacunas e registre horário e estação usados. Até isso existir, Open-Meteo permanece como fallback automático e deve ser identificado como estimativa de modelo.

## Limitações e próximos passos

- A aplicação reporta a maior saída do classificador, não uma probabilidade calibrada de ocorrência. `f1_macro` escolhe o modelo, mas não transforma os valores de `predict_proba` em risco calibrado.
- A avaliação atual em `evaluate_models` usa divisão aleatória de linhas, e o `TRAINING_DATA` já pode conter observações sintéticas antes dessa divisão. Cópias/variações sintéticas podem aparecer nos dois lados e inflar métricas. Antes de uso operacional, separar por tempo ou município/estação, aplicar balanceamento apenas no treino e avaliar calibração e desempenho por desastre.
- A ligação município-estação é feita pela estação mais próxima em distância de coordenadas; não considera relevo, bacias hidrográficas ou cobertura de radar.
- Open-Meteo é uma fonte de prototipagem não comercial, sujeita aos termos e limites publicados pelo serviço. Confirmar licença, atribuição e disponibilidade antes de publicar ou usar comercialmente.
- Um alerta real deve considerar limiares oficiais, validação com Defesa Civil, monitoramento contínuo, histórico auditável e comunicação clara de incerteza; o protótipo não deve ser usado para decisões de emergência.