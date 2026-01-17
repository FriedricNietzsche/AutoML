class ReporterAgent:
    def __init__(self):
        self.reports = []

    def generate_report(self, training_results):
        report = {
            "metrics": self.extract_metrics(training_results),
            "visualizations": self.create_visualizations(training_results),
            "summary": self.create_summary(training_results)
        }
        self.reports.append(report)
        return report

    def extract_metrics(self, training_results):
        # Extract relevant metrics from training results
        metrics = {
            "accuracy": training_results.get("accuracy"),
            "f1_score": training_results.get("f1_score"),
            "rmse": training_results.get("rmse"),
            "r2": training_results.get("r2")
        }
        return metrics

    def create_visualizations(self, training_results):
        # Create visualizations based on training results
        visualizations = {
            "loss_curve": self.plot_loss_curve(training_results),
            "confusion_matrix": self.plot_confusion_matrix(training_results)
        }
        return visualizations

    def plot_loss_curve(self, training_results):
        # Placeholder for loss curve plotting logic
        return "URL to loss curve visualization"

    def plot_confusion_matrix(self, training_results):
        # Placeholder for confusion matrix plotting logic
        return "URL to confusion matrix visualization"

    def create_summary(self, training_results):
        # Create a summary of the training results
        summary = f"Training completed with accuracy: {training_results.get('accuracy')}"
        return summary

    def get_reports(self):
        return self.reports