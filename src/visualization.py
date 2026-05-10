# visualization.py
# Author: Bernadette Burks
# Created: Sept. 14, 2025

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    display = None
    clear_output = None

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import symptom_index
from src.training import predict_disease, rf_model, nb_model, svm_model



# Example usage
print(predict_disease("skin_rash, fever, headache", rf_model, nb_model, svm_model))

symptom_selector = widgets.SelectMultiple(
    options=list(symptom_index.keys()),  # symptom_index must already exist
    description='Symptoms:',
    rows=10,
    layout=widgets.Layout(width='50%')
)

# Dynamic output area
output = widgets.Output()

def update_visualization(change):
    with output:
        clear_output(wait=True)  # Clears old output for smooth updates
        
        selected_symptoms = ', '.join(symptom_selector.value)
        if not selected_symptoms:
            print("Please select at least one symptom to see predictions.")
            return

        # Get predictions
        results = predict_disease(selected_symptoms, rf_model, nb_model, svm_model) # pyright: ignore[reportUndefinedVariable]

        # Print textual results
        print("Selected Symptoms:", selected_symptoms)
        print("Predictions:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        # Prepare dynamic bar chart
        labels = list(results.keys())
        final_pred = results["Final Prediction"]
        values = [1 if results[key] == final_pred else 0 for key in labels]
        colors = ['green' if v == 1 else 'orange' for v in values]

        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            text=[f"{results[label]}" for label in labels],
            hoverinfo='text+y',
            marker_color=colors
        )])

        fig.update_layout(
            title=f"Prediction Agreement for: {selected_symptoms}",
            yaxis=dict(title='Match with Final Prediction', tickvals=[0,1], ticktext=['No','Yes']),
            xaxis=dict(title='Model'),
            showlegend=False,
            template='plotly_dark',
            height=450
        )
        fig.show()

        symptom_selector.observe(update_visualization, names='value')

        display(symptom_selector, output)