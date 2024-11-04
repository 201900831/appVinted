
from dash import Dash, html, dcc
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from cargar_datosFinal2 import load_and_prepare_data
from ML_models import run_models
from ML_models import train_model
from ML_models import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize the Dash app
app = Dash(__name__, title="Customer Purchase Prediction")

# Load and prepare the data
file_path = "online_shoppers_intention.csv"  # Replace with the path to your actual CSV file
df = load_and_prepare_data(file_path)

# Run models and get evaluation results
results_df = run_models(file_path)

# Create a bar chart to show the count of purchases by VisitorType
visitor_purchase_fig = px.histogram(df, x="VisitorType", color="will_buy", barmode="group",
                                    title="Purchases by Visitor Type",
                                    labels={"VisitorType": "Visitor Type", "will_buy": "Purchase Made (1=Yes, 0=No)"},
                                    category_orders={"will_buy": [0, 1]})

# Create a bar chart for model performance metrics (Precision, Recall, F1 Score, and ROC AUC)
performance_metrics_fig = px.bar(
    results_df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1 Score', 'ROC AUC']),
    x="Model", y="value", color="variable",
    title="Model Performance Metrics",
    labels={"value": "Score", "variable": "Metric"},
    barmode="group"
)

# Generate a confusion matrix for one of the models (e.g., Random Forest)
# For simplicity, we will use Random Forest as an example
X = df.drop(columns=["will_buy"])
y = df["will_buy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model for confusion matrix example
pipeline_rf = train_model(Pipeline(steps=[
    ('preprocessor', preprocess_data(X_train)),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
]), X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

# Create a Plotly heatmap for the confusion matrix
confusion_matrix_fig = ff.create_annotated_heatmap(
    z=cm,
    x=['Predicted No Purchase', 'Predicted Purchase'],
    y=['Actual No Purchase', 'Actual Purchase'],
    colorscale='Blues',
    showscale=True
)
confusion_matrix_fig.update_layout(title_text="Confusion Matrix for Random Forest Classifier")

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Customer Purchase Prediction Dashboard'),

    html.Div(children='''
        This app visualizes customer purchase predictions and model evaluation metrics.
    '''),

    # Display the purchase distribution chart
    dcc.Graph(
        id='visitor-purchase-distribution',
        figure=visitor_purchase_fig
    ),

    # Display model performance metrics
    dcc.Graph(
        id='model-performance-metrics',
        figure=performance_metrics_fig
    ),

    # Display the confusion matrix
    dcc.Graph(
        id='confusion-matrix',
        figure=confusion_matrix_fig
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
