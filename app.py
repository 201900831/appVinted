from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
from src.cargar_datosFinal2 import load_and_prepare_data
from src.ML_models import run_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Initialize the Dash app
app = Dash(__name__, title="Customer Purchase and Model Evaluation Dashboard")

# Load and prepare the data
file_path = "online_shoppers_intention.csv"  # Replace with your actual CSV file
df = load_and_prepare_data(file_path)

# Run models and get evaluation results
results_df = run_models(file_path)

# Train Random Forest for feature importance
X = df.drop(columns=["will_buy"])
y = df["will_buy"]
categorical_cols = X.select_dtypes(include=['category', 'object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
pipeline_rf = RandomForestClassifier(random_state=42)
pipeline_rf.fit(preprocessor.fit_transform(X), y)

# Feature Importance
feature_importances = pipeline_rf.feature_importances_
features = preprocessor.get_feature_names_out()
feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)

from sklearn.preprocessing import OrdinalEncoder

# Create a copy of the dataframe to preprocess
df_encoded = df.copy()

# Identify categorical variables
categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns

# Apply ordinal encoding
encoder = OrdinalEncoder()
df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])
# Compute the correlation matrix
correlation_matrix = df_encoded.corr()


# Create a more readable heatmap
correlation_fig = px.imshow(
    correlation_matrix,
    labels=dict(x="Features", y="Features", color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    title="Correlation Matrix of Features",
    color_continuous_scale="RdBu_r",  # High-contrast color scale
    text_auto=True,  # Annotate with correlation values
)

# Customize figure for better clarity
correlation_fig.update_layout(
    title_font_size=20,
    title_x=0.5,
    xaxis_tickangle=45,
    height=800,  # Increase figure height
    width=800,  # Increase figure width
)

# Create confusion matrix (example using Random Forest model)
conf_matrix = [[300, 20], [10, 170]]
conf_matrix_fig = ff.create_annotated_heatmap(
    z=conf_matrix,
    x=["Predicted No", "Predicted Yes"],
    y=["Actual No", "Actual Yes"],
    colorscale="Blues",
    showscale=True,
)
conf_matrix_fig.update_layout(
    title="Confusion Matrix (Example)",
    xaxis_title="Predicted",
    yaxis_title="Actual",
)

# Layout of the app
app.layout = html.Div(
    children=[
        html.H1(
            "Customer Purchase and Model Evaluation Dashboard",
            style={
                "textAlign": "center",
                "color": "#ffffff",
                "backgroundColor": "#2a3f5f",
                "padding": "20px",
                "borderRadius": "5px",
                "marginBottom": "20px",
            },
        ),
        html.P(
            "This application provides insights into customer purchase behavior "
            "and evaluates the performance of predictive models. Explore data and results below.",
            style={
                "textAlign": "center",
                "fontSize": "16px",
                "color": "#333333",
                "marginBottom": "30px",
            },
        ),
        
        # Section 1: Purchase Distribution
        html.Div(
            [
                html.H2("Purchase Distribution", style={"color": "#2a3f5f", "marginBottom": "10px"}),
                html.P(
                    "Select a category to visualize the distribution of purchases. This helps identify trends in customer behavior "
                    "across different visitor types, months, or regions.",
                    style={"fontSize": "14px", "color": "#666666", "marginBottom": "10px"},
                ),
                html.Label("Select a Category to Visualize Purchases:", style={"fontWeight": "bold", "color": "#333333"}),
                dcc.Dropdown(
                    id="category-dropdown",
                    options=[
                        {"label": "Visitor Type", "value": "VisitorType"},
                        {"label": "Month", "value": "Month"},
                        {"label": "Region", "value": "Region"},
                    ],
                    value="VisitorType",
                    style={"marginBottom": "20px"},
                ),
                dcc.Graph(id="purchase-distribution"),
            ],
            style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9", "marginBottom": "30px"},
        ),

        # Section: Correlation Matrix
        html.Div(
            [
                html.H2("Correlation Matrix", style={"color": "#2a3f5f", "marginBottom": "10px"}),
                html.P(
                    "This matrix shows the correlation between numerical features in the dataset. "
                    "High positive or negative correlations indicate strong relationships between variables."
                    "In this case, it is important to see which are the variables that have higher correlations with the will_buy variable",
                    style={"fontSize": "14px", "color": "#666666", "marginBottom": "10px"},
                ),
                dcc.Graph(id="correlation-matrix", figure=correlation_fig),
            ],
            style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9", "marginBottom": "30px"},
        ),
        

        # Section 3: Variable Distribution
        html.Div(
            [
                html.H2("Variable Distribution", style={"color": "#2a3f5f", "marginBottom": "10px"}),
                html.P("Select a continuous variable to visualize its distribution among customers."
                       "This chart shows how selected variables, such as Bounce Rates, Exit Rates, and Product Related Duration, correlate with purchase behavior, providing insights to optimize user engagement.", style={"fontSize": "14px", "color": "#666666", "marginBottom": "10px"}),
                dcc.Dropdown(
                    id="histogram-dropdown",
                    options=[
                        {"label": "Bounce Rates", "value": "BounceRates"},
                        {"label": "Exit Rates", "value": "ExitRates"},
                        {"label": "Product Related Duration", "value": "ProductRelated_Duration"},
                    ],
                    value="BounceRates",
                    style={"marginBottom": "20px"},
                ),
                dcc.Graph(id="variable-histogram"),
            ],
            style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9", "marginBottom": "30px"},
        ),

        # Section 4: Model Evaluation Metrics
        html.Div(
            [
                html.H2("Model Evaluation Metrics", style={"color": "#2a3f5f", "marginBottom": "10px"}),
                html.P(
                    "Use the dropdown to explore various metrics that evaluate the performance of the machine learning model that will predict if a customer entering a website will buy or not. "
                    "This includes the confusion matrix, F1 Score, Precision, Recall, and ROC Curve.",
                    style={"fontSize": "14px", "color": "#666666", "marginBottom": "10px"},
                ),
                html.Label("Select a Metric to Visualize:", style={"fontWeight": "bold", "color": "#333333"}),
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[
                        {"label": "Confusion Matrix", "value": "conf_matrix"},
                        {"label": "F1 Score", "value": "f1_score"},
                        {"label": "Precision and Recall", "value": "precision_recall"},
                        {"label": "ROC Curve", "value": "roc_curve"},
                    ],
                    value="conf_matrix",
                    style={"marginBottom": "20px"},
                ),
                dcc.Graph(id="model-metric"),
            ],
            style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9"},
        ),

        # Section 5: Feature Importance
        html.Div(
            [
                html.H2("Feature Importance", style={"color": "#2a3f5f", "marginBottom": "10px"}),
                html.P(
                    "This visualization will help with understanding which are the most relevant features that will determine if a customer will buy or not, according to the model selected",
                    style={"fontSize": "14px", "color": "#666666", "marginBottom": "10px"},
                ),
                dcc.Graph(
                    id="feature-importance",
                    figure=px.bar(
                        feature_importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance",
                        labels={"Feature": "Feature", "Importance": "Importance"}
                    ),
                ),
            ],
            style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "5px", "backgroundColor": "#f9f9f9", "marginBottom": "30px"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "900px", "margin": "0 auto", "backgroundColor": "#e8ebef", "padding": "20px"},
)

@app.callback(
    Output("purchase-distribution", "figure"),
    Input("category-dropdown", "value"),
)

def update_purchase_distribution(selected_category):
    fig = px.histogram(
                df,
        x=selected_category,
        color="will_buy",
        barmode="group",
        title=f"Purchases by {selected_category}",
        labels={selected_category: selected_category, "will_buy": "Purchase Made (1=Yes, 0=No)"},
        category_orders={"will_buy": [0, 1]},
    )
    return fig


@app.callback(
    Output("variable-histogram", "figure"),
    Input("histogram-dropdown", "value"),
)
def update_variable_histogram(selected_variable):
    fig = px.histogram(
        df,
        x=selected_variable,
        color="will_buy",
        barmode="group",
        title=f"Distribution of {selected_variable}",
        labels={selected_variable: selected_variable, "will_buy": "Purchase Made (1=Yes, 0=No)"},
    )
    return fig


@app.callback(
    Output("model-metric", "figure"),
    Input("metric-dropdown", "value"),
)
def update_model_metric(selected_metric):
    if selected_metric == "conf_matrix":
        return conf_matrix_fig
    elif selected_metric == "f1_score":
        return px.bar(
            results_df,
            x="Model",
            y="F1 Score",
            title="F1 Score by Model",
            labels={"F1 Score": "F1 Score", "Model": "Model"},
            color_discrete_sequence=["#636EFA"],
        )
    elif selected_metric == "precision_recall":
        return px.bar(
            results_df.melt(id_vars="Model", value_vars=["Precision", "Recall"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="Precision and Recall by Model",
            labels={"value": "Score", "variable": "Metric"},
        )
    elif selected_metric == "roc_curve":
        return px.line(
            x=[0, 0.1, 0.3, 0.6, 1],
            y=[0, 0.4, 0.7, 0.9, 1],
            title="ROC Curve (Example)",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        )
    return {}


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
