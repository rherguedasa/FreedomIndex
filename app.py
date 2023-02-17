#--------------------------------------------------------------------------------------------------IMPORTAMOS LIBRERIAS---------------------------------------------------------------------------------------------


# librerias básicas
import numpy as np 
import pandas as pd
from math import sqrt

# ML
import sklearn as sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# importamos librerias graficación 
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = 'plotly_dark'
import folium

# quitamos warnings que nos puedan aparecer en las celdas
import warnings
warnings.filterwarnings('ignore')

# importamos librerias streamlit
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from streamlit.elements.utils import (check_callback_rules, check_session_state_rules, get_label_visibility_proto_value)
from geopy.geocoders import Nominatim

# translate 
from googletrans import Translator


#-----------------------------------------------------------------------------------------------------------DATAFRAMES QUE VAMOS A IMPORTAR----------------------------------------------------------------------------------------------------------------------


# leemos el dataframe que vamos a usar-
df = pd.read_csv('freedomeconomicindex.csv')
df2= pd.read_csv('mldfeconomic.csv')
values = pd.read_excel('index2022.xls')
# los dataframes ya están procesados, para ver el preprocesamiento usar el notebook


#------------------------------------------------------------------------------------------------------------------COMENZAMOS LA APP ----------------------------------------------------------------------------------------------------------------------------------------------


st.set_page_config(page_title='ÍNDICE DE LIBERTAD ECONÓMICA', layout='wide' , page_icon="", initial_sidebar_state="expanded",)
st.title('ÍNDICE DE LIBERTAD ECONÓMICA')
st.header('ANÁLISIS EXPLORATORIO Y MODELO PREDICTIVO')

# establecemos la imagen de fondo de la app
# además añadimos codigo CSS para eliminar el fondo
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
        .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("background.png")

# configuración para evitar la visualización de un warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# creamos la side bar
st.sidebar.write('---')
st.sidebar.title('¿QUÉ QUIERES VER?')
index = st.sidebar.selectbox('Selecciona una página', 
                            ('INTRODUCCIÓN', 
                            'ANÁLISIS POR REGIONES', 
                            'ANÁLISIS CORRELACIONAL',
                            'MODELO PREDICTIVO', 
                            'CONCLUSIONES'))
st.sidebar.write('---')
st.markdown(
    f"""
    <style>
    [data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0);
    }}
    [data-testid="stSidebar"]{{                 
    background-color: rgba(0, 0, 0, 0);
    border: 0.5px solid #ff4b4b;
        }}
    """
, unsafe_allow_html=True)


#--------------------------------------------------------------------------------------------------------------------------INTRODUCCIÓN----------------------------------------------------------------------------------------------------------------------------------------


if index == 'INTRODUCCIÓN':
    introduccion = st.tabs(['INTRODUCCIÓN','DATASET'])
    st.markdown(
        f"""
        <style>  
        [data-baseweb="tab"] {{
        background-color: rgba(0, 0, 0, 0);
            }}
        </style>
        """
    , unsafe_allow_html=True)
    

    # añadimos una página para ver de donde hemos sacado el dataset y aplicamos una pequeña introducción
    tab_plots = introduccion[0]
    with tab_plots:
        st.header('¿Qué es el Índice de libertad Económica de la fundación Heritage?')
        st.markdown("""**El Índice de libertad Económica de la fundación Heritage mide la libertad económica en diferentes sectores, centrándose en el estado de derecho, las limitaciones del gobierno, 
                    la eficiencia regulatoria y la apertura de mercados. The Heritage Foundation lo realiza anualmente desde 1995, donde en su última versión se midió para 186 países**.""")
        st.markdown("""**Este índice tiene un sesgo político alto, pero el estudio solo valorará las variables macroeconómicas que se miden en él**.""")
        st.write(f'<iframe src="https://www.heritage.org/index/heatmap" width="1200" height="800" style="overflow:auto"></iframe>', unsafe_allow_html=True)
        url1 = "https://www.heritage.org/index/"
        st.markdown("[Heritage Foundation](%s)" % url1)

    # mostramos el dataframe extraido de Heritage Index
    tab_plots = introduccion[1]
    with tab_plots:
        st.header('DataSet')
        st.markdown('**DataSet extraído de la web Heritage, tras el preprocesamiento**')
        st.write(df, width = 1200)
        url3="https://www.heritage.org/index/excel/2019/index2019_data.xls"
        st.markdown("[Descarga el archivo raw](%s)" % url3)


#----------------------------------------------------------------------------------------------------------------SEGUNDA PESTAÑA: ANÁLISIS POR REGIONES-------------------------------------------------------------------------------------------------------------------------------


elif index == 'ANÁLISIS POR REGIONES':
    regiones = st.tabs(['PAÍSES A ANALIZAR','DISTRIBUCIÓN DE LAS REGIONES EN EL MUNDO','DATOS MACRO EN CADA REGIÓN'])
    st.markdown(
    f"""
    <style>  
    [data-baseweb="tab"] {{
    background-color: rgba(0, 0, 0, 0);
    }}
    </style>
    """
    , unsafe_allow_html=True)

    # añadimos un mapa interactivo para ver de dónde hemos sacado los mapas en la primera pestaña
    tab_plots = regiones[0]
    with tab_plots:
        st.header('¿Cuántos países se analizan?')
        with st.expander("**Países dentro del índice**"):
            st.write("**Se analizan un total de 186 países y a la vez  las 5 regiones que las engloban.**")
        # ploteamos el mapa de folium previamente guardado en HTML
        html = open("map.html", "r", encoding='utf-8').read()
        st.components.v1.html(html, width=1200, height=800)

    # mostramos los países agrupados por regiones en la segunda pestaña
    tab_plots = regiones[1]
    with tab_plots:
        st.header('Países agrupados por regiones')
        with st.expander("**% de regiones analizados dentro del mundo**"):
            st.write("**La mayoría del territorio ocupado por los países analizados se distribuye entre África (35,8%) y Asia (23,1), siendo Europa la menor parte del territorio mundial (24.2%).**")
        # hacemos un group by de las regiones y los paises y los contabilizamos
        grouped = df.groupby('Region')
        counts = grouped['Country'].count()
        result = counts.reset_index()
        # añadimos los colores y creamos el pie
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        fig = go.Figure(data=go.Pie(values=result['Country'], labels=result['Region'], marker=dict(colors=colors)))
        fig.update_layout(width=1200,height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # mostramos un subplot sobre datos macro de las regiones en la tercera pestaña       
    tab_plots = regiones[2]
    with tab_plots:
        st.header('Datos macroeconómicos por regiones')
        with st.expander('**Análisis de la población, inflación, desempleo y deuda pública**'):
            st.write("**-La mayoría de la población se concentra en Asia**")
            st.write("**-Las regiones con mayor inflación son la zona Subsahariana y América**")
            st.write("**-Europa y la zona de África Subsahariana son las regiones con más desempleo**")
            st.write("**-Además Europa y Africa Subsahariana son las zonas con una deuda pública mayor (% del PBI)**")
        # creamos un groupby de varias variables macroeconomicas y la variable 'Region'
        grouped_df = df.groupby(['Region'])['Population (Millions)'].sum().reset_index()
        grouped_df_inflation = df.groupby(['Region'])['Inflation (%)'].sum().reset_index()
        grouped_df_unemployment = df.groupby(['Region'])['Unemployment (%)'].sum().reset_index()
        grouped_df_public_debt = df.groupby(['Region'])['Public Debt (% of GDP)'].sum().reset_index()
        # añadimos los cores
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        # creamos el subplot
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Population (Millions) by Region","Inflation (%) by Region", "Unemployment (%) by Region","Public Debt (% of GDP) by Region" ))
        # añadimos el primer subplot 
        fig.add_trace(go.Bar(x=grouped_df['Region'], y=grouped_df['Population (Millions)'], marker=dict(color=colors), name="Population (Millions)"), row=1, col=1)
        fig.update_yaxes(title_text="Population (Millions)", row=1, col=1)
        # añadimos el segundo subplot 
        fig.add_trace(go.Bar(x=grouped_df_inflation['Region'], y=grouped_df_inflation['Inflation (%)'], marker=dict(color=colors), name="Inflation (%)"), row=1, col=2)
        fig.update_yaxes(title_text="Inflation (%)", row=1, col=2)
        # añadimos el tercer subplot 
        fig.add_trace(go.Bar(x=grouped_df_unemployment['Region'], y=grouped_df_unemployment['Unemployment (%)'], marker=dict(color=colors), name="Unemployment (%)"),row=2, col=1)
        fig.update_yaxes(title_text="Unemployment (%)", row=2, col=1)
        # # añadimos el cuarto subplot 
        fig.add_trace(go.Bar(x=grouped_df_public_debt['Region'], y=grouped_df_public_debt['Public Debt (% of GDP)'], marker=dict(color=colors), name="Public Debt (% of GDP)"), row=2, col=2)
        fig.update_yaxes(title_text="Public Debt (% of GDP)", row=2, col=2)
        # añadimos un update para ciertos paramétros que quedan por añadir
        fig.update_layout(width = 1200, height = 1000, showlegend=True)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig)


#----------------------------------------------------------------------------------------------------------------TERCERA PESTAÑA: ANÁLISIS CORRELACIONAL------------------------------------------------------------------------------------------------------------------


elif index == 'ANÁLISIS CORRELACIONAL':
    cor = st.tabs(['CORRELACIONES','RANKING POR PAISES / REGIONES' ,'2019 SCORE', 'LIBERTAD EMPRESARIAL', 'LIBERTAD DE INVERSIÓN', 'PRODUCTO INTERIOR BRUTO'])
    st.markdown(
    f"""
    <style>  
    [data-baseweb="tab"] {{
    background-color: rgba(0, 0, 0, 0);
    }}
    </style>
    """
    , unsafe_allow_html=True)

    # primera pestaña con la matriz de correlación
    tab_plots = cor[0]

    with tab_plots:
        # creamos la variable corr con las columnas que se nos pide realizando una correlación de Pearson (r = cov(X, Y) / (std(X) * std(Y))
        corr = df[['Country', 'Region', 'World Rank', 'Region Rank', '2019 Score',
                'Property Rights', 'Judical Effectiveness', 'Government Integrity',
                'Tax Burden', "Gov't Spending", 'Fiscal Health', 'Business Freedom',
                'Labor Freedom', 'Monetary Freedom', 'Trade Freedom',
                'Investment Freedom', 'Financial Freedom', 'Tariff Rate (%)',
                'Income Tax Rate (%)', 'Corporate Tax Rate (%)', 'Tax Burden % of GDP',
                "Gov't Expenditure % of GDP", 'Population (Millions)',
                'GDP (Billions, PPP)$', 'GDP Growth Rate (%)',
                '5 Year GDP Growth Rate (%)', 'GDP per Capita (PPP)$',
                'Unemployment (%)', 'Inflation (%)', 'FDI Inflow (Millions)',
                'Public Debt (% of GDP)', 'Latitude', 'Longitude']].corr()
        # creamos un gráfico de calor utilizando la matriz de correlación 
        st.header('Matriz de Correlación')
        with st.expander("**Correlaciones dentro de esta matriz**"):
            st.write("**Utilizando el método de Pearson (r = cov(X, Y) / (std(X) * std(Y)), se escogen las variables que tienen un ≥ 0.70 para este análisis**")
        # creamos un gráfico de calor utilizando la matriz de correlación
        fig = px.imshow(corr, color_continuous_scale=px.colors.sequential.Jet)
        fig.update_layout(title ='Pearson correlation matrix' )
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(width=1200,  height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # segunda pestaña con la matriz de correlación
    tab_plots = cor[1]
    with tab_plots:
        st.header('Ranking mundial por países y regiones')
        with st.expander("**Diferencias entre el ranking mundial**"):
            st.write("""**El ranking se da en función de la puntuación del Score de 2019, varía dependiendo si es mundial o por regiones. 
                    Es decir, un país tendrá una puntuación diferente comparado con todos los países o solo comparado con los de su región**""")
        # hacemos un subplot de una comparativa de el Worl Rank con el Region Rank 
        fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], row_heights=[0.6], 
                            specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]], 
                            subplot_titles=("World Rank 2019 Freedom Index","Region Rank 2019 Freedom Index"))
        # añadimos el primer subplot y sus características
        fig.add_trace(
            go.Scattergeo(lon = df['Longitude'], lat = df['Latitude'], text = df['Country'] + ' (' + df['World Rank'].astype(str) + ')', mode = 'markers',
                        marker = dict(size = 20, sizemode = 'diameter', color = df['World Rank'], colorscale = 'Plasma', reversescale = False),
                        name = 'World Rank'), row=1, col=1)
        # añadimos las características del plot
        fig.update_layout(
                title = 'World Rank 2019 Freedom Index',
                geo = dict(
                scope = 'world',
                showland = True,
                landcolor = 'White',
                showcountries = True,
                countrycolor = 'Black',
                showocean=True,
                oceancolor="LightBlue",
                lakecolor="LightBlue",
                projection = dict(type = "orthographic", scale=1.5),
                lonaxis = dict(range = [-270, 270]),
                lataxis = dict(range = [-270, 270])
            ),
        )
        fig.update_xaxes(domain=[0, 0.45], row=1, col=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_layout(width = 1200, height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        # añadimos el segundo subplot y sus características
        fig.add_trace(
            go.Scattergeo(lon = df['Longitude'], lat = df['Latitude'], text = df['Country'] + ' (' + df['Region Rank'].astype(str) + ')', mode = 'markers',
                        marker = dict(size = 20, sizemode = 'diameter', color = df['Region Rank'], colorscale = 'Plasma', reversescale = False),
                        name = 'Region Rank'), row=1, col=2)
        # añadimos las características del plot
        fig.update_layout(
                title = 'Region Rank 2019 Freedom Index',
                geo2 = dict(
                scope = 'world',
                showland = True,
                landcolor = 'White',
                showcountries = True,
                countrycolor = 'Black',
                showocean=True,
                oceancolor="LightBlue",
                lakecolor="LightBlue",
                projection = dict(type = "orthographic", scale=1.5),
                lonaxis = dict(range = [-270, 270]),
                lataxis = dict(range = [-270, 270])
            ),
        )
        #añadimos las características del plot
        fig.update_xaxes(domain=[0.55, 1], row=1, col=2)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_layout(width = 1200, height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # tercera pestaña con el 2019 Score
    tab_plots = cor[2]
    with tab_plots:
        st.header('2019 Score')
        st.markdown('**El ranking puntúa datos entre 0 y 100, donde 0 significa «ausencia de libertad económica» y 100 significa «libertad económica total».**')
        st.markdown('**Hay 12 aspectos divididos en cuatro categorías:**')
        st.markdown('-**Estado de Derecho(Rule of Law):** derechos de propiedad, integridad de gobierno, eficiencia judicial.')
        st.markdown('-**Tamaño de Gobierno:** carga fiscal, gasto de gobierno, salud fiscal.')
        st.markdown('-**Eficiencia Regulatoria:** facilidad de hacer negocios, libertad de trabajo, libertad de política monetaria.')
        st.markdown('-**Apertura de Mercados (Free Trade):** libre comercio, libertad de inversión y libertad financiera.')
        with st.expander("**¿Qué países lideran el ranking del 2019 Freedom Index?**"):
            st.write("""**El 2019 Index Score, lo lideran países como Hong Kong, Singapur y Nueva Zelanda, 
                    mientras que el Score más bajo lo tienen países como Cuba, Venezuela y en último lugar Corea del Norte**""")
            # hacemos un top 20 de los mejores y peores paises valorados según el 2019 Score
            top_20_countries = df.sort_values(by='2019 Score', ascending=False).head(20)
            bottom_20_countries = df.sort_values(by='2019 Score', ascending=True).head(20)
            # creamos dos columnas para poder hacer dos subplots y comparlaos
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Top 20 Countries by 2019 Score","Bottom 20 Countries by 2019 Score"))
            colorscale = 'Viridis'
            fig.add_trace(go.Bar(x=top_20_countries['Country'], y=top_20_countries['2019 Score'], name='Top 20', marker_color=top_20_countries['2019 Score'], marker_colorscale=colorscale), 1, 1)
            fig.add_trace(go.Bar(x=bottom_20_countries['Country'], y=bottom_20_countries['2019 Score'], name='Bottom 20', marker_color=bottom_20_countries['2019 Score'], marker_colorscale=colorscale), 1, 2)
            fig.update_layout(
                xaxis=dict(title='Top 20 Countries by 2019 Score'),
                xaxis2=dict(title='Bottom 20 Countries by 2019 Score'),
                yaxis=dict(title='2019 Score'),
                showlegend=True, 
                width=1200,
                height=800)
            fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig)

    with tab_plots:
        # hacemos un plot del 2019 Score por ciudades con un Scatter geo
        fig = go.Figure(data=go.Scattergeo(
            lon = df['Longitude'],
            lat = df['Latitude'],
            text = df['Country'] + ' (' + df['2019 Score'].astype(str) + ')',
            mode = 'markers',
            marker = dict(
                size = 20,
                sizemode = 'diameter',
                color = df['2019 Score'],
                colorscale = 'Jet',
                showscale = True,
                reversescale = False
                ),
            ))
        # añadimos la forma de globo terráqueo y además lo personalizamos
        fig.update_layout(
            geo = dict(
                scope = 'world',
                showland = True,
                landcolor = 'White',
                showcountries = True,
                countrycolor = 'Black',
                showocean=True,
                oceancolor="LightBlue",
                lakecolor="LightBlue",
                projection = dict(type = "orthographic", scale=1.5),
                lonaxis = dict(range = [-270, 270]),
                lataxis = dict(range = [-270, 270])
            ),
        )
        fig.update_layout(width = 1200, height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

        st.write('---')

        st.header('Variables dependientes del gobierno y del mercado')
        with st.expander("**¿Qué variables encontramos correlacionadas (≥ 0.70) con el 2019 Score?**"):
            st.write("""**Hay 8 variables que tienen una correlación positiva con el 2019 Score.
                    Se relacionan además con variables dependientes del gobierno y variables dependientes de las finanzas o el mercado del país**""")
        # creamos un scatterplot para todas las variables relacionadas con el 2019 Score
            fig = go.Figure(data=[go.Scatter(x=df['Property Rights'], y=df['2019 Score'], mode='markers', marker=dict(symbol='square'), name='Property Rights', text=df['Country']),
                    go.Scatter(x=df['Judical Effectiveness'], y=df['2019 Score'], mode='markers', marker=dict(symbol='triangle-up'), name='Judical Effectiveness', text=df['Country']),
                    go.Scatter(x=df['Government Integrity'], y=df['2019 Score'], mode='markers', marker=dict(symbol='circle'), name='Government Integrity', text=df['Country']),
                    go.Scatter(x=df['Tax Burden'], y=df['2019 Score'], mode='markers', marker=dict(symbol='cross'), name='Tax Burden', text=df['Country']),
                    go.Scatter(x=df['Business Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='x'), name='Business Freedom', text=df['Country']),
                    go.Scatter(x=df['Trade Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='pentagon'), name='Trade Freedom', text=df['Country']),
                    go.Scatter(x=df['Investment Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='star'), name='Investment Freedom', text=df['Country']),
                    go.Scatter(x=df['Financial Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='triangle-down'), name='Financial Freedom', text=df['Country'])],
                layout=go.Layout(title='Relationship between the 2019 score and its dependent variables.',
                                xaxis_title='Government/Finance-dependent variables',
                                yaxis_title='2019 Score',
                                showlegend=True))
            fig.update_layout(width = 1200, height=800)
            fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)

        # creamos un subplot para todas las variables relacionadas con el 2019 Score y las variables gubernamentales
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"},{"type": "scatter"}]], subplot_titles=("Government dependent variables","Finance dependent variables") )
        fig.add_trace(go.Scatter(x=df['Property Rights'], y=df['2019 Score'], mode='markers', marker=dict(symbol='square'), name='Property Rights', text=df['Country']), row = 1, col = 1)
        fig.add_trace(go.Scatter(x=df['Judical Effectiveness'], y=df['2019 Score'], mode='markers', marker=dict(symbol='triangle-up'), name='Judical Effectiveness', text=df['Country']), row = 1, col = 1)
        fig.add_trace(go.Scatter(x=df['Government Integrity'], y=df['2019 Score'], mode='markers', marker=dict(symbol='circle'), name='Government Integrity', text=df['Country']), row = 1, col = 1)
        fig.add_trace(go.Scatter(x=df['Tax Burden'], y=df['2019 Score'], mode='markers', marker=dict(symbol='cross'), name='Tax Burden', text=df['Country']), row = 1, col = 1)
        # añadimos un update para mejorar la visualización
        fig.update_layout(title='Relationship between the 2019 score and Goverment dependent variables.',
                                        xaxis_title='Government dependent variables',
                                        yaxis_title='2019 Score',
                                        showlegend=True, 
                                        width=1600,
                                        height=700)
        # creamos un subplot para todas las variables relacionadas con el 2019 Score y las variables relacionadas con las finanzas
        fig.add_trace(go.Scatter(x=df['Business Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='x'), name='Business Freedom', text=df['Country']), row = 1, col = 2)
        fig.add_trace(go.Scatter(x=df['Trade Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='pentagon'), name='Trade Freedom', text=df['Country']), row = 1, col = 2)
        fig.add_trace(go.Scatter(x=df['Investment Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='star'), name='Investment Freedom', text=df['Country']), row = 1, col = 2)
        fig.add_trace(go.Scatter(x=df['Financial Freedom'], y=df['2019 Score'], mode='markers', marker=dict(symbol='triangle-down'), name='Financial Freedom', text=df['Country']), row = 1, col = 2)
        # añadimos un update para mejorar la visualización
        fig.update_layout(
            title='Relationship between the 2019 score and dependent variables',
            xaxis=dict(title='Government dependent variables', showgrid=True, zeroline=True, anchor='y2'),
            xaxis2=dict(title='Finance dependent variables', showgrid=True, zeroline=True, anchor='y2'),
            yaxis=dict(title='2019 Score', showgrid=True, zeroline=True),
            showlegend=True, 
            width=1200,
            height=600)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)


    # hacemos un plot relacionando la variable business freedom con otras variables gubernamentales por países
    tab_plots = cor[3]
    with tab_plots:
        st.header('Libertad empresarial')
        with st.expander("**¿Qué variables encontramos correlacionadas con la libertad empresarial**"):
            st.write("""**La libertad empresarial esta estrechamente relacionada con el 2019 Score, los derechos de propiedad, la eficacia judicial y la integridad del gobierno, 
            siendo los países mejor puntuados en el 2019 Score los que tienen mayor libertad empresarial.**""")
        size = df['Business Freedom'] / df['Business Freedom'].mean() * 20 + 5 #  toma el valor de "Business Freedom" para cada país y lo normaliza dividiéndolo por la media de todos los valores de "Business Freedom". Luego, se multiplica por 20 para obtener un valor adecuado para el tamaño del marcador y se agrega 5 para asegurarse de que los marcadores sean lo suficientemente grandes para ser visibles en el gráfico
        fig = go.Figure()
        for col in ['2019 Score','Property Rights', 'Judical Effectiveness', 'Government Integrity']:
            fig.add_scatter(x=df[col], y=df['Business Freedom'], name=col, mode='markers', text=df['Country'],
                                marker=dict(size=size, sizemode='diameter'))
        fig.update_layout(xaxis_title="2019 Score, Property Rights, Judical Effectiveness, Government Integrity",
                            yaxis_title="Business Freedom" , width=1200, height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # hacemos un plot relacionando Investmen Freedom  con Financial Freedom por países
    tab_plots = cor[4]
    with tab_plots:
        st.header('Libertad de invesión: relación con la libertad financiera por países')
        with st.expander("**¿Qué países tienen mayor libertad para invertir?**"):
            st.write("""**Los países mejor rankeados tienen una mayor libertad de inversión y por lo tanto una mayor libertad financiera**""")
        fig = go.Figure()
        fig.add_scatter(x=df['Financial Freedom'], y=df['Investment Freedom'], mode='markers',
                text=df['Country'], marker=dict(size=df['Investment Freedom'], sizemode='diameter'),
                marker_color=df['Investment Freedom'], marker_colorscale='Plasma')
        fig.update_layout(xaxis_title="Financial Freedom", yaxis_title="Investment Freedom", width=1200, height=800)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # relación de el PBI con la población y el fluje de inversión extranjera
    tab_plots = cor[5]
    with tab_plots:
        st.header('Relación del PIB con la población y la inversión extranjera')
        with st.expander("**¿Qué variables se relacionan con el PIB?**"):
            st.write("""**Los países con mayor población tienen un PIB mayor, además las primeras potencias mundiales reciben mayor flujo de inversión extranjera**""")
        fig = go.Figure()
        for col, color, name in zip(['Population (Millions)', 'FDI Inflow (Millions)'], ['#19D3F3', '#FF6692'], ['Population (Millions)', 'FDI Inflow (Millions)']):
            fig.add_trace(go.Scatter(
                x=df[col],
                y=df['GDP (Billions, PPP)$'],
                mode='markers',
                marker=dict(
                    size=10*(df['GDP (Billions, PPP)$'] / df['GDP (Billions, PPP)$'].max()),
                    sizemode='diameter',
                    sizeref=0.1,
                    color=color,
                    reversescale=True,
                    opacity=0.8
                ),
                text=df['Country'],
                name=name
            ))
        fig.update_layout(
            width=1200,
            height=600,
            xaxis_title="'Population (Millions)'/'FDI Inflow (Millions)'",
            yaxis_title="GDP (Billions, PPP)$")
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)


#----------------------------------------------------------------------------------------------------------------CUARTA PESTAÑA: MODELO PREDICTIVO------------------------------------------------------------------------------------------------------------------


elif index == 'MODELO PREDICTIVO':
    ml = st.tabs(['MODELO PREDICTIVO','COMPARATIVA REGRESIONES'])
    st.markdown(
    f"""
    <style>  
    [data-baseweb="tab"] {{
    background-color: rgba(0, 0, 0, 0);
    }}
    </style>
    """
    , unsafe_allow_html=True)

    # primera pestaña con la matriz de correlación
    tab_plots = ml[0]
    with tab_plots:
        st.header('Modelo predictivo: Compara los valores predichos con el año 2022')
        # seleccionamos las variables predictoras
        X = df2[['Property Rights', 'Judical Effectiveness', 'Government Integrity', 'Tax Burden', 'Business Freedom', 'Trade Freedom', 'Investment Freedom', 'Financial Freedom']]
        # seleccionamos las variables independientes
        y = df2['2019 Score']
        # realizamos una sepración de nuestro dataset con un train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

        # realizamos una Regresión Lineal Múltiple
        lin = LinearRegression().fit(X_train, y_train)
        # hacemos predicciones utilizando el modelo de regresión en el conjunto de pruebas
        y_pred = lin.predict(X_test)
        # creamos un dataframe con los datos reales y las predicciones
        df_linear = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_lin = mean_absolute_error(y_test, y_pred)
        mse_lin = mean_squared_error(y_test, y_pred)
        rmse_lin = sqrt(mse_lin)
        r2_lin = r2_score(y_test, y_pred)
        rmsle_lin = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_lin = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # realizamos una Regresión Lineal Múltiple con Ridge Regression
        rid = Ridge(alpha=1.0).fit(X_train, y_train)
        # Hacer predicciones utilizando el modelo de regresión en el conjunto de pruebas
        y_pred = rid.predict(X_test)
        # creamos un dataframe con los datos reales y las predicciones
        df_ridge = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_rid = mean_absolute_error(y_test, y_pred)
        mse_rid = mean_squared_error(y_test, y_pred)
        rmse_rid = sqrt(mse_rid)
        r2_rid = r2_score(y_test, y_pred)
        rmsle_rid = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_rid = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # realiamos una regresión lasso
        las = Lasso(alpha=0.1).fit(X_train, y_train)
        # hacemos las predicciones de nuestro test
        y_pred = las.predict(X_test)
        # creamos un dataframe para guardar estas predicciones
        df_lasso = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_las = mean_absolute_error(y_test, y_pred)
        mse_las = mean_squared_error(y_test, y_pred)
        rmse_las = sqrt(mse_las)
        r2_las = r2_score(y_test, y_pred)
        rmsle_las = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_las = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # creamos una lista con los nombres de los países
        countries = values['Country'].unique()

        # creamos la interfaz de usuario
        st.subheader('Predicción del Índice de Libertad')
        country = st.text_input('Escribe el nombre de del país en inglés [Ej: Spain]')

        if country in countries:
            # buscamos los datos del país seleccionado
            country_data = values[values['Country'] == country][['Property Rights', 'Judical Effectiveness',
                'Government Integrity', 'Tax Burden', 'Business Freedom',
                'Trade Freedom', 'Investment Freedom ', 'Financial Freedom']]
            actual_value = values[values['Country'] == country]['2022 Score'].values[0]
        
            # realizamos  las predicciones
            predictionlin = lin.predict(country_data)

            # creamos un objeto de traductor
            translator = Translator()

            # traducimos el nombre del país al castellano
            country_spanish = translator.translate(country, dest='es').text

            # creamos el mensaje con el nombre del país traducido
            mensaje = "**El Índice de libertad económica predicho para {} es {} y la comparativa con el valor actual del mismo índice es {}**".format(country_spanish, round(predictionlin[0], 2), actual_value)

            # mostramos el mensaje con st.write()
            st.write(mensaje)

            # obtenemos las coordenadas del país
            geolocator = Nominatim(user_agent="myapp")
            location = geolocator.geocode(country)
            latitude = location.latitude
            longitude = location.longitude

            # creamos un mapa con la ubicación del país
            map = folium.Map(location=[latitude, longitude], zoom_start=5)
            folium.Marker(location=[latitude, longitude], popup=country).add_to(map)
            
            # guardamos el mapa en un archivo HTML
            map.save('map1.html')
            with open('map1.html', 'r') as f:
                html = f.read()
            st.components.v1.html(html, width=1200, height=600)
        else:
            if country:
                st.write('Porfavor, escriba el nombre del país que quiere predecir en inglés y usando una mayúscula al principio')

    tab_plots = ml[1]
    # creamos el código para mostrar lo que hemos hecho en cada uno de nuestras regresiones
    with tab_plots:
        st.header('Comparativa Regresión Lineal, Ridge y Lasso')
        st.subheader('Selección de las varibles y división del dataset en train y test')
        code = """# seleccionamos las variables predictoras
        X = df2[['Property Rights', 'Judical Effectiveness', 'Government Integrity', 'Tax Burden', 'Business Freedom', 'Trade Freedom', 'Investment Freedom', 'Financial Freedom']]
        # seleccionamos las variables independientes
        y = df2['2019 Score']
        # realizamos una sepración de nuestro dataset con un train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        """
        st.code(code, language='python')

        st.subheader('Regresión Lineal')
        code = """ # Realizar una Regresión Lineal Múltiple
        lin = LinearRegression().fit(X_train, y_train)
        # Hacer predicciones utilizando el modelo de regresión en el conjunto de pruebas
        y_pred = lin.predict(X_test)
        # Crear un dataframe con los datos reales y las predicciones
        df_linear = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_lin = mean_absolute_error(y_test, y_pred)
        mse_lin = mean_squared_error(y_test, y_pred)
        rmse_lin = sqrt(mse_lin)
        r2_lin = r2_score(y_test, y_pred)
        rmsle_lin = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_lin = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        """
        st.code(code, language='python')

        st.subheader('Regresión Ridge')
        code = """# Realizar una Regresión Lineal Múltiple con Ridge Regression
        rid = Ridge(alpha=1.0).fit(X_train, y_train)
        # Hacer predicciones utilizando el modelo de regresión en el conjunto de pruebas
        y_pred = rid.predict(X_test)
        # Crear un dataframe con los datos reales y las predicciones
        df_ridge = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_rid = mean_absolute_error(y_test, y_pred)
        mse_rid = mean_squared_error(y_test, y_pred)
        rmse_rid = sqrt(mse_rid)
        r2_rid = r2_score(y_test, y_pred)
        rmsle_rid = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_rid = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        """
        st.code(code, language='python')

        st.subheader('Regresión Lasso')
        code = """# realiamos una regresión lasso
        las = Lasso(alpha=0.1).fit(X_train, y_train)
        # hacemos las predicciones de nuestro test
        y_pred = las.predict(X_test)
        # creamos un dataframe para guardar estas predicciones
        df_lasso = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        # calculamos las métricas de error
        mae_las = mean_absolute_error(y_test, y_pred)
        mse_las = mean_squared_error(y_test, y_pred)
        rmse_las = sqrt(mse_las)
        r2_las = r2_score(y_test, y_pred)
        rmsle_las = sqrt(mean_squared_log_error(y_test, y_pred))
        mape_las = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        """
        st.code(code, language='python')

        st.write('---')

        st.subheader('Coeficiente de cada modelo')
        # extraemos los coeficientes de cada modelo y crear un DataFrame
        coef_df = pd.DataFrame({
            'Variables': X.columns,
            'Linear Regression': lin.coef_,
            'Ridge Regression': rid.coef_,
            'Lasso Regression': las.coef_,
            })
        coef_df = coef_df.melt(id_vars='Variables', var_name='Model', value_name='Coefficient')
        # creamos una figura con múltiples subplots, uno para cada modelo
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Linear Regression","Ridge Regression", "Lasso Regression"))
        # se itera sobre cada modelo y agregar una línea para cada variable predictora
        for i, model in enumerate(coef_df['Model'].unique()):
            model_coef_df = coef_df[coef_df['Model'] == model]
            fig.add_trace(go.Scatter(x=model_coef_df['Variables'], y=model_coef_df['Coefficient'], name=model), row=i+1, col=1)
        # update del diseño de la figura
        fig.update_layout(height=1600, width=1200)
        fig.update_yaxes(title_text='Coeficiente', tickformat='.3f')
        fig.update_xaxes(title_text='Variable predictora')
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

        st.write('---')

        st.subheader('Comportamiento de los modelos de regresión')
        # creamos una figura con múltiples subplots, uno para cada modelo
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Linear Regression","Ridge Regression", "Lasso Regression"))
        # iteramos sobre cada modelo y agregar una línea para cada variable predictora
        for i, model in enumerate([lin, rid, las]):
            # hacemos predicciones en el conjunto de prueba
            y_pred = model.predict(X_test)
            # agregamos una gráfica de dispersión con las predicciones y los valores reales
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='#EF553B'), showlegend=False), row=i+1, col=1)
            # agregamos una línea diagonal que representa la línea de predicción perfecta
            fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode='lines', line=dict(color='#636EFA'), showlegend=False), row=i+1, col=1)
            # update del título de cada subfigura con el nombre del modelo
            fig.update_yaxes(title_text='Predicción', tickformat='.2f', row=i+1, col=1)
            fig.update_xaxes(title_text='Valor real', tickformat='.2f', row=i+1, col=1)
            fig.update_layout(height=1600, width=1200)
        # agregamos una leyenda para cada modelo
        fig.update_layout(legend=dict(y=0.99, x=0.01))
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

        st.write('---')

        st.subheader("Métricas de error por modelo de regresión")
        # creamos las trazas de datos para cada modelo de regresión
        traces = [
            go.Scatter(
                x=['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'],
                y=[mae_lin, mse_lin, rmse_lin, r2_lin, rmsle_lin, mape_lin],
                mode='markers',
                marker=dict(size=14, color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']),
                name='Regresión lineal múltiple'
            ),
            go.Scatter(
                x=['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'],
                y=[mae_rid, mse_rid, rmse_rid, r2_rid, rmsle_rid, mape_rid],
                mode='markers',
                marker=dict(size=14, color=['#17BECF', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']),
                name='Ridge Regression'
            ),
            go.Scatter(
                x=['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'],
                y=[mae_las, mse_las, rmse_las, r2_las, rmsle_las, mape_las],
                mode='markers',
                marker=dict(size=14, color=['#990099', '#0099C6', '#DD4477', '#66AA00', '#B82E2E', '#316395']),
                name='Lasso Regression'
            ),]
        # creamos una figura con las trazas de datos
        fig = go.Figure(data=traces)
        # agregamos títulos y etiquetas de ejes
        fig.update_layout(
            xaxis_title="Métrica",
            yaxis_title="Valor",
            width = 1200)
        fig.update_layout(margin={"r":80,"t":80,"l":80,"b":80})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
        st.markdown('-**Los tres modelos tienen un MAE bastante similar, alrededor de 3.66, lo que significa que las predicciones tienen un error promedio absoluto de aproximadamente 3.66 unidades de la variable dependiente.**')
        st.markdown('-**También tienen un MSE y RMSE bastante similar, con valores alrededor de 22.91 y 4.79, respectivamente. Esto sugiere que el error de las predicciones del modelo no es demasiado grande, pero no es tan pequeño como el error MAE.**')
        st.markdown('-**En cuanto a R2, los tres modelos tienen valores cercanos a 0.88, lo que indica que el modelo es capaz de explicar alrededor del 88% de la variabilidad en la variable dependiente. Este es un buen indicador de que el modelo es adecuado para el conjunto de datos.**')
        st.markdown('-**El modelo Lasso tiene un RMSLE ligeramente más alto que los otros dos modelos, lo que sugiere que puede tener más dificultades para predecir valores en las partes extremas del rango de valores.**')
        st.markdown('-**El MAPE del modelo Lasso es el más alto, con un valor de alrededor del 10.10, lo que indica que las predicciones tienen un error promedio absoluto porcentual de alrededor del 10.10%. Los otros dos modelos tienen un MAPE ligeramente más bajo, con un valor de alrededor del 10.05%.**')
        st.markdown('-**Los tres modelos son bastante similares en términos de su capacidad para predecir los valores de la variable dependiente, pero el modelo de Regresión Lineal y el modelo de Ridge parecen ser ligeramente superiores al modelo de Lasso.**')


#----------------------------------------------------------------------------------------------------------------QUINTA PESTAÑA: CONCLUSIONES------------------------------------------------------------------------------------------------------------------


elif index == 'CONCLUSIONES':
    st.header('Conclusiones')
    st.markdown('- **Los países con mayores niveles de libertad económica tienden a atraer más inversiones extranjeras. Esto se debe a que los inversores buscan países donde puedan invertir con confianza y sin obstáculos regulatorios excesivos**')
    st.markdown('- **El índice de libertad económica fomenta el crecimiento económico, ya que se encuentran mejor posicionados frente a un sistema capitalista o de libre mercado, lo que a su vez puede generar empleo, aumentar los ingresos y mejorar el bienestar de la población en general.**')
    st.markdown('- **Los países con mayores niveles de libertad económica suelen tener un PBI per cápita más alto. Esto se debe a que la libertad económica fomenta el emprendimiento, la innovación y la inversión, lo que a su vez puede llevar a un aumento en la producción y el ingreso por habitante.**')
    st.markdown('- **Hong Kong, Singapur y Nueva Zelanda son los países con mayor libertad económica, según este índice. Estos países tienen economías abiertas y liberales, con bajos niveles de intervención gubernamental.**')
    st.markdown('- **Los países de América Latina tienen, en general, bajos niveles de libertad económica. Solo Chile y Uruguay se sitúan en el top 50 de países con mayor libertad económica. La mayoría de los países latinoamericanos tienen economías más cerradas y reguladas.**')
    st.markdown('- **Los países con mayores niveles de libertad económica tienden a tener mayores niveles de prosperidad económica. Esto se debe a que la libertad económica fomenta la inversión, el emprendimiento y la innovación, lo que a su vez lleva a un crecimiento económico sostenible.**')
    st.markdown('- **Los países con mayores niveles de libertad económica también tienden a tener mayores niveles de bienestar social. Esto se debe a que la libertad económica fomenta la creación de empleo y la reducción de la pobreza, lo que a su vez mejora la calidad de vida de la población.**')


#------------------------------------------------------------------------------------------------------------------------------END-------------------------------------------------------------------------------------------------------------------------------------------------------------
