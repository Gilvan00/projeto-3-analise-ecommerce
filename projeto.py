import pandas as pd
import plotly.express as px
import numpy as np

from dash import Dash, html, dcc


# 1. Ler o arquivo CSV
df = pd.read_csv('ecommerce_estatistica.csv')


# 2. Criar os gráficos

# Gráfico 1: Histograma dos preços
fig1 = px.histogram(
    df,
    x='Preço',
    nbins=20,
    title='Distribuição dos Preços dos Produtos',
    labels={'Preço': 'Preço', 'count': 'Frequência'}
)


# Gráfico 2: Dispersão entre preço e nota
fig2 = px.scatter(
    df,
    x='Preço',
    y='Nota',
    title='Preço x Notas dos Produtos',
    labels={'Preço': 'Preço', 'Nota': 'Nota'},
    opacity=0.6
)


# Gráfico 3: Mapa de calor das correlações
corr = df[['Nota', 'N_Avaliações', 'Desconto', 'Preço']].corr()

fig3 = px.imshow(
    corr,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    title='Mapa de Calor das Correlações'
)


# Gráfico 4: As 10 marcas com mais produtos
marcas = df['Marca'].value_counts().head(10).sort_values()

fig4 = px.bar(
    x=marcas.values,
    y=marcas.index,
    orientation='h',
    title='Top 10 Marcas com Mais Produtos',
    labels={'x': 'Quantidade de produtos', 'y': 'Marca'}
)


# Gráfico 5: Distribuição dos produtos por gênero
generos = df['Gênero'].value_counts().head(5)

fig5 = px.pie(
    names=generos.index,
    values=generos.values,
    title='Distribuição de Produtos por Gênero',
    hole=0.04
)


# Gráfico 6: Densidade dos preços
fig6 = px.violin(
    df,
    x='Preço',
    box=False,
    points=False,
    title='Distribuição da Densidade dos Preços'
)

fig6.update_traces(
    orientation='h',
    side='positive',
    width=3
)

fig6.update_layout(
    xaxis_title='Preço',
    yaxis_title='Densidade'
)


# Gráfico 7: Regressão entre avaliações e preço
dados_reg = df[['N_Avaliações', 'Preço']].dropna()

fig7 = px.scatter(
    dados_reg,
    x='N_Avaliações',
    y='Preço',
    title='Regressão: Número de Avaliações x Preço',
    opacity=0.5
)

# Separar os valores de X e Y
x = dados_reg['N_Avaliações'].to_numpy()
y = dados_reg['Preço'].to_numpy()

# Calcular a reta de regressão linear
coeficientes = np.polyfit(x, y, 1)
reta = np.poly1d(coeficientes)

x_linha = np.linspace(x.min(), x.max(), 100)

fig7.add_scatter(
    x=x_linha,
    y=reta(x_linha),
    mode='lines',
    name='Regressão linear'
)


# 3. Criar a aplicação Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1('Análise Estatística do E-commerce'),

    html.P('Visualização interativa dos dados dos produtos.'),

    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig6),
    dcc.Graph(figure=fig7)
])


# 4. Executar a aplicação
if __name__ == '__main__':
    app.run(debug=True, port=8050)
