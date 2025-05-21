import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial de titulo dashboard
st.set_page_config(page_title="Análisis de Ventas - Tiendas de Conveniencia", layout="wide")
st.title("Dashboard de Análisis de Ventas")

# Cargar datos, limpieza de columnas irrelevantes, transformación(tiempo y fecha) y conversión de categorías
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop(['Invoice ID', 'gross margin percentage'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
    df['Day'] = df['Date'].dt.day_name()
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    df['Month'] = df['Date'].dt.month
    # Conversión de columnas categóricas
    for col in ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Day']:
        df[col] = df[col].astype('category')
    return df

df = load_data()

# Análisis exploratorio inicial
st.subheader("Análisis Exploratorio del Dataset")

# Expander para agrupar y ocultar si se desea
with st.expander("Ver análisis exploratorio"):
    
    st.write("**Primeras filas del dataset:**")
    st.dataframe(df.head())

    st.write("**Resumen de columnas:**")
    st.write(df.columns.tolist())

    st.write("**Información del dataframe:**")
    buffer = df.info(buf=None)
    # Capturar la salida de df.info()
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("**Estadísticas descriptivas:**")
    st.dataframe(df.describe(include='all'))

# Filtros
st.sidebar.header("Filtros")
cities = st.sidebar.multiselect("Seleccionar Ciudad", options=df['City'].unique(), default=df['City'].unique())
customer_types = st.sidebar.multiselect("Tipo de Cliente", options=df['Customer type'].unique(), default=df['Customer type'].unique())
product_lines = st.sidebar.multiselect("Línea de Producto", options=df['Product line'].unique(), default=df['Product line'].unique())

filtered_df = df[
    (df['City'].isin(cities)) &
    (df['Customer type'].isin(customer_types)) &
    (df['Product line'].isin(product_lines))
]
# MÉTRICAS RESUMEN
st.subheader("Métricas Clave del Segmento Seleccionado")

# Calcular métricas
total_ventas = filtered_df['Total'].sum()
facturas = filtered_df.shape[0]
clientes_unicos = filtered_df['Customer type'].count()  # Aquí asumimos que cada fila es una transacción única
ticket_promedio = total_ventas / facturas if facturas > 0 else 0
rating_promedio = filtered_df['Rating'].mean()

# Mostrar métricas en 3 columnas
col1, col2, col3 = st.columns(3)

col1.metric("Total Ventas", f"${total_ventas:,.2f}")
col2.metric("Nº de Facturas", f"{facturas}")
col3.metric("Ticket Promedio", f"${ticket_promedio:,.2f}")

# Segunda fila de métricas
col4, col5 = st.columns(2)
col4.metric("Calificación Promedio", f"{rating_promedio:.2f} / 10")

# Producto más vendido
producto_top = (
    filtered_df.groupby('Product line')['Total'].sum()
    .sort_values(ascending=False)
    .idxmax()
)
col5.metric("Producto Más Vendido", producto_top)

# Visualizaciones
# Sección 1: Ventas por fecha
st.subheader("Evolución de Ventas Totales")
sales_by_date = filtered_df.groupby('Date')['Total'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(12, 4))
sns.lineplot(data=sales_by_date, x='Date', y='Total', ax=ax1)
ax1.set_title("Ventas Totales por Fecha")
st.pyplot(fig1)

# Sección 2: Ingreso por línea de producto
st.subheader("Ingresos por Línea de Producto")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(data=filtered_df, x='Product line', y='Total', estimator=sum, ci=None, ax=ax2)
ax2.set_title("Total Vendido por Línea de Producto")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)

# Sección 3: Distribución de Calificaciones
st.subheader("Distribución de Calificaciones")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.histplot(filtered_df['Rating'], bins=10, kde=True, ax=ax3)
st.pyplot(fig3)

# Sección 4: Comparación de gasto por tipo de cliente
st.subheader("Gasto por Tipo de Cliente")
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.boxplot(data=filtered_df, x='Customer type', y='Total', ax=ax4)
st.pyplot(fig4)

# Sección 5: COGS vs Ingreso Bruto
st.subheader("Relación entre Costo (COGS) e Ingreso Bruto")
fig5, ax5 = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=filtered_df, x='cogs', y='gross income', hue='Customer type', ax=ax5)
st.pyplot(fig5)

# Sección 6: Métodos de pago preferidos
st.subheader("Métodos de Pago")
fig6, ax6 = plt.subplots(figsize=(8, 4))
sns.countplot(data=filtered_df, x='Payment', order=filtered_df['Payment'].value_counts().index, ax=ax6)
st.pyplot(fig6)

# Sección 7: Mapa de correlación
st.subheader("Matriz de Correlación Numérica")
fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax7)
st.pyplot(fig7)

# Sección 8: 
# Ingreso bruto por sucursal y línea de producto
st.subheader("Ingreso Bruto por Sucursal y Línea de Producto")

# Agrupamos los datos
grouped = filtered_df.groupby(['Branch', 'Product line'])['Total'].sum().reset_index()

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=grouped,
    x='Branch',
    y='Total',
    hue='Product line',
    palette='Set2',
    ax=ax
)

ax.set_title('Ingreso Total por Sucursal y Línea de Producto')
ax.set_ylabel('Ingreso ($)')
ax.set_xlabel('Sucursal')
ax.legend(title='Línea de Producto', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
st.pyplot(fig)
