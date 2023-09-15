import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os

# Load the dataset
df = pd.read_csv("iris.csv")

# Choose only the columns I want
available_columns = [col for col in df.columns if col != 'target']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Title
    html.H1("Iris Species Prediction", style={'textAlign': 'center'}),

    # Subtitle 1
    html.H2("1. Analyze column distributions through the use of histograms:", style={'textAlign': 'left'}),

    # Dropdown menu to select a column
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in available_columns],
        value=available_columns[0]
    ),

    # Data Manipulation Section
    html.H2("2. Data Manipulation:", style={'textAlign': 'left'}),

    # Checkbox for normalizing the selected feature
    html.Label(id='normalize-label', children="Normalize the selected feature:"),
    dcc.Checklist(
        id='normalize-checkbox',
        options=[{'label': 'Normalize', 'value': 'normalize'}],
        value=[]
    ),


    # Checkbox for removing outliers
    html.Label(id='outliers-label', children="Remove outliers based on a defined threshold:"),
    dcc.Checklist(
        id='outliers-checkbox',
        options=[{'label': 'Remove Outliers', 'value': 'outliers'}],
        value=[]
    ),

    # Message when species column is chosen
    html.Div(id='message', style={'color': 'red', 'marginTop': '10px'}),

    # Histogram
    dcc.Graph(id='histogram'),

    # Subtitle 3
    html.H2("3. Choose the model and Predict:", style={'textAlign': 'left'}),

    # Dropdown menu to select a model
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'SVM Model - (train accuracy of 97% and Test accuracy of 100%)', 'value': 'svm_model'},
            {'label': 'Logistic Regression Model - (train accuracy of 96% and Test accuracy of 100%)', 'value': 'logistic_reg_model'},
            {'label': 'Decision Tree Model - (train accuracy of 100% and Test accuracy of 100%)', 'value': 'dt_model'},
            {'label': 'K-NN Model - (train accuracy of 95% and Test accuracy of 100%)', 'value': 'knn_model'},
            {'label': 'Gaussian Naive Bayes Model - (train accuracy of 94% and Test accuracy of 100%)', 'value': 'gnb_model'},
        ],
        value='svm_model'  # Default model selection
    ),

    # Empty Space
    html.P("", style={'textAlign': 'right', 'marginRight': '10px'}),

    # Input boxes for custom input values
    html.Label("Sepal Length:"),
    dcc.Input(id='sepal-length', type='number', value=0),
    html.Label("Sepal Width:"),
    dcc.Input(id='sepal-width', type='number', value=0),
    html.Label("Petal Length:"),
    dcc.Input(id='petal-length', type='number', value=0),
    html.Label("Petal Width:"),
    dcc.Input(id='petal-width', type='number', value=0),

    # Empty Space
    html.P("", style={'textAlign': 'right', 'marginRight': '10px'}),

    # Button to trigger prediction
    html.Button('Predict', id='predict-button', n_clicks=0),

    # Placeholder to display the predicted output
    html.Div(id='prediction-output', style={'marginTop': '10px'}),

    # Data scientist's name in the lower right corner
    html.P("Data Scientist: Simon Pierre Charbel", style={'textAlign': 'right', 'marginRight': '10px'}),

])

# Callback to update the histogram based on column selection and data manipulations
@app.callback(
    Output('histogram', 'figure'),
    Output('normalize-label', 'children'),
    Output('outliers-label', 'children'),
    Output('normalize-checkbox', 'options'),
    Output('outliers-checkbox', 'options'),
    Output('message', 'children'),
    Input('column-dropdown', 'value'),
    Input('normalize-checkbox', 'value'),
    Input('outliers-checkbox', 'value')
)
def update_histogram(selected_column, normalize, outliers):
    data = df[selected_column]
    normalize_options = [{'label': 'Normalize', 'value': 'normalize'}]
    outliers_options = [{'label': 'Remove Outliers', 'value': 'outliers'}]

    # Check if the selected column is 'species'
    if selected_column == 'species':
        # Disable checkboxes, change labels, and display a message
        normalize_options = []
        outliers_options = []
        normalize_label = "Features unavailable !"
        outliers_label = ""
        message = "'Normalization' and 'Outlier Removal' options are not applicable to the 'species' column."
    else:
        message = ""
        normalize_label = "Normalize the selected feature:"
        outliers_label = "Remove outliers based on a defined threshold:"

        # Data Manipulations
        if 'normalize' in normalize:
            # Normalize each feature (column) between 0 and 1
            col_min = df[available_columns].min()
            col_max = df[available_columns].max()
            df[available_columns] = (df[available_columns] - col_min) / (col_max - col_min)

        if 'outliers' in outliers:
            z_scores = (data - data.mean()) / data.std()
            data = data[(z_scores >= -3) & (z_scores <= 3)]  # Remove outliers

    filtered_df = df[df[selected_column].isin(data)]  # Filter the DataFrame after normalization

    fig = px.histogram(filtered_df, x=selected_column, color='species',
                       color_discrete_map={'setosa': 'lightblue', 'versicolor': 'blue', 'virginica': 'darkblue'})

    return fig, normalize_label, outliers_label, normalize_options, outliers_options, message


# Callback to predict the species and display it
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('model-dropdown', 'value'),
    Input('sepal-length', 'value'),
    Input('sepal-width', 'value'),
    Input('petal-length', 'value'),
    Input('petal-width', 'value')
)
def predict_species(n_clicks, selected_model, sepal_length, sepal_width, petal_length, petal_width):
    if n_clicks is not None and n_clicks > 0:
        # Define the models directory
        models_directory = "C:/Users/simon/AppData/Roaming/JetBrains/PyCharmCE2022.2/scratches/models/"

        # Load the selected model
        model_path = os.path.join(models_directory, f'{selected_model}.pkl')
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Create a new data point with custom input values
        new_data_point = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])

        # Use the model to make predictions
        predicted_class_label = loaded_model.predict(new_data_point)[0]  # Get the predicted class label

        class_labels = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
        predicted_species = class_labels[predicted_class_label]  # Map to species name

        return html.Div([
            html.P(f'Predicted Species : {predicted_species} ({predicted_class_label})', style={'fontWeight': 'bold'}),
        ])
    else:
        return html.Div([html.P(f'No Prediction Made.')])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
