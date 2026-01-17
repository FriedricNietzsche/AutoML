def export_model(model, preprocessing_pipeline, model_name, export_dir):
    import joblib
    import os

    # Create the export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Export the model
    model_path = os.path.join(export_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    # Export the preprocessing pipeline
    pipeline_path = os.path.join(export_dir, f"{model_name}_pipeline.joblib")
    joblib.dump(preprocessing_pipeline, pipeline_path)

    return model_path, pipeline_path


def export_notebook(notebook_content, export_dir, model_name):
    import os

    # Create the export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Define the notebook path
    notebook_path = os.path.join(export_dir, f"{model_name}.ipynb")

    # Write the notebook content to a file
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)

    return notebook_path


def export_report(report_data, export_dir, model_name):
    import os
    import json

    # Create the export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Define the report path
    report_path = os.path.join(export_dir, f"{model_name}_report.json")

    # Write the report data to a JSON file
    with open(report_path, 'w') as f:
        json.dump(report_data, f)

    return report_path