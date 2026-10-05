# Sistema de alerta e classificação desastres

Sistema desenvolvido como trabalho da matéria de Planejamento e Gestão de Projetos, no segundo semestre de 2026 na Universidade Federal da Fronteira Sul - Campus Chapecó.

Utiliza bases abertas de informações sobre desastres brasileiros para treinar um modelo de machine learning, e comparar com dados meteorológicos atuais e retornar a possibilidade de enxurradas, alagamentos, inundações e deslizamentos.

Alunos: Abimael Mendes e Felipe Daniel Nerling.

## Como executar o main

1. Abra o terminal na pasta do projeto.
2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Abra o notebook principal em `src/main.ipynb` no VS Code ou no Jupyter.
5. Execute todas as células do notebook para carregar os dados, treinar os modelos e gerar as previsões.

Se preferir executar via linha de comando, também é possível rodar o notebook com:
```bash
jupyter notebook src/main.ipynb
```

## Aplicação Streamlit

1. Execute todas as células de `src/main.ipynb`. A avaliação seleciona pelo maior `f1_macro` e exporta o modelo retreinado em `models/best_model.joblib`.
2. Inicie a interface na raiz do projeto:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. Passe o mouse sobre um município para ver o nome e clique no polígono para carregar a consulta de risco estimado.

O mapa desenha diretamente os objetos SVG da malha municipal do IBGE, sem biblioteca cartográfica, basemap ou tiles. As variáveis meteorológicas são obtidas do Open-Meteo com cache de 12 horas. Na página, baixe o modelo existente ou gere/exporte um novo se o arquivo não estiver no deploy. Consulte [docs/streamlit.md](docs/streamlit.md) para arquitetura, limitações e próximos passos.

## Fonte meteorológica do INMET

O catálogo oficial de estações automáticas do INMET está disponível em `https://apitempo.inmet.gov.br/estacoes/T` e informa códigos, coordenadas e situação operacional; ele pode ser usado para localizar a estação `Operante` mais próxima de cada município. As observações recentes da tabela oficial usam uma rota protegida por reCAPTCHA, portanto não estão integradas ao app como uma API de servidor sem autenticação. Por essa limitação, o app usa o Open-Meteo como fonte automática; detalhes e limitações estão em [docs/streamlit.md](docs/streamlit.md).

## Bases de dados
- INMET - Informações meteorológicas (https://portal.inmet.gov.br/dadoshistoricos)
- S2ID - Histórico de desastres (https://s2id.mi.gov.br/paginas/relatorios/)
- IBGE - Informações dos municípios (https://www.ibge.gov.br/estatisticas/multidominio/ciencia-tecnologia-e-inovacao/27385-localidades.html)