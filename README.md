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

## Bases de dados
- INMET - Informações meteorológicas (https://portal.inmet.gov.br/dadoshistoricos)
- S2ID - Histórico de desastres (https://s2id.mi.gov.br/paginas/relatorios/)
- IBGE - Informações dos municípios (https://www.ibge.gov.br/estatisticas/multidominio/ciencia-tecnologia-e-inovacao/27385-localidades.html)