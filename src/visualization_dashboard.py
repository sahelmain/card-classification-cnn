import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

class ModelVisualizationDashboard:
    """
    Comprehensive visualization dashboard for model performance analysis.
    Features confusion matrices, training plots, and model comparisons.
    """
    
    def __init__(self, results_dir='results/visualizations'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style for professional-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_training_history(self, history, model_name="Model"):
        """Create comprehensive training history visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Training & Validation Accuracy
        ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'{model_name} - Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training & Validation Loss
        ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title(f'{model_name} - Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance Summary
        final_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        best_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_acc) + 1
        
        summary_text = f"""
        Final Validation Accuracy: {final_acc:.4f}
        Final Validation Loss: {final_loss:.4f}
        Best Validation Accuracy: {best_acc:.4f}
        Best Epoch: {best_epoch}
        Total Epochs: {len(epochs)}
        """
        
        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax3.set_title(f'{model_name} - Performance Summary', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Epoch vs Loss trend
        ax4.plot(epochs, history.history['val_loss'], 'g-', linewidth=2)
        ax4.set_title(f'{model_name} - Validation Loss Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_comparison(self, model_results):
        """Create comprehensive model comparison dashboard"""
        models = list(model_results.keys())
        
        # Create interactive comparison chart
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Accuracy Comparison', 'Precision vs Recall', 'F1-Score Comparison',
                          'Training Time', 'Model Complexity', 'Performance vs Time'),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Extract data
        accuracies = [model_results[model]['accuracy'] for model in models]
        precisions = [model_results[model]['precision'] for model in models]
        recalls = [model_results[model]['recall'] for model in models]
        f1_scores = [model_results[model]['f1_score'] for model in models]
        times = [model_results[model]['training_time'] for model in models]
        params = [model_results[model].get('parameters', 0) for model in models]
        
        # Accuracy comparison
        fig.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', 
                            marker_color='lightblue'), row=1, col=1)
        
        # Precision vs Recall scatter
        fig.add_trace(go.Scatter(x=precisions, y=recalls, mode='markers+text',
                               text=models, textposition="top center",
                               marker=dict(size=12, color='red'),
                               name='Precision vs Recall'), row=1, col=2)
        
        # F1-Score comparison
        fig.add_trace(go.Bar(x=models, y=f1_scores, name='F1-Score',
                            marker_color='lightgreen'), row=1, col=3)
        
        # Training time
        fig.add_trace(go.Bar(x=models, y=times, name='Training Time (s)',
                            marker_color='orange'), row=2, col=1)
        
        # Model complexity (parameters)
        fig.add_trace(go.Bar(x=models, y=params, name='Parameters',
                            marker_color='purple'), row=2, col=2)
        
        # Performance vs Time scatter
        fig.add_trace(go.Scatter(x=times, y=accuracies, mode='markers+text',
                               text=models, textposition="top center",
                               marker=dict(size=12, color='darkblue'),
                               name='Accuracy vs Time'), row=2, col=3)
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Comprehensive Model Performance Comparison")
        fig.write_html(f'{self.results_dir}/model_comparison_dashboard.html')
        fig.show()
        
    def create_performance_report(self, model_results):
        """Generate comprehensive performance report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Card Classification Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; }}
                .best {{ background-color: #2ecc71; color: white; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎴 Card Classification Model Performance Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Model Performance Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Training Time (s)</th>
                    <th>Parameters</th>
                </tr>
        """
        
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        
        for model, metrics in model_results.items():
            row_class = 'best' if model == best_model else ''
            report_html += f"""
                <tr class="{row_class}">
                    <td><strong>{model}</strong></td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1_score']:.4f}</td>
                    <td>{metrics['training_time']:.2f}</td>
                    <td>{metrics.get('parameters', 'N/A'):,}</td>
                </tr>
            """
        
        report_html += f"""
            </table>
            
            <div class="metric best">
                <h3>🏆 Best Performing Model: {best_model}</h3>
                <p>Accuracy: {model_results[best_model]['accuracy']:.4f}</p>
            </div>
            
            <h2>🎯 Key Insights</h2>
            <ul>
                <li>Total models evaluated: {len(model_results)}</li>
                <li>Best accuracy achieved: {max([m['accuracy'] for m in model_results.values()]):.4f}</li>
                <li>Fastest training: {min([m['training_time'] for m in model_results.values()]):.2f}s</li>
            </ul>
            
        </body>
        </html>
        """
        
        with open(f'{self.results_dir}/performance_report.html', 'w') as f:
            f.write(report_html)
        
        print(f"📊 Performance report saved to {self.results_dir}/performance_report.html")

# Example usage demonstration
def demonstrate_dashboard():
    """Demonstrate the visualization dashboard capabilities"""
    dashboard = ModelVisualizationDashboard()
    
    # Sample model results for demonstration
    sample_results = {
        'Custom CNN': {
            'accuracy': 0.814,
            'precision': 0.812,
            'recall': 0.816,
            'f1_score': 0.814,
            'training_time': 2400.0,
            'parameters': 134000000
        },
        'Lightweight CNN': {
            'accuracy': 0.740,
            'precision': 0.738,
            'recall': 0.742,
            'f1_score': 0.740,
            'training_time': 189.4,
            'parameters': 4364117
        },
        'MobileNetV2': {
            'accuracy': 0.517,
            'precision': 0.520,
            'recall': 0.515,
            'f1_score': 0.517,
            'training_time': 145.8,
            'parameters': 2800000
        }
    }
    
    # Create comparison dashboard
    dashboard.plot_model_comparison(sample_results)
    dashboard.create_performance_report(sample_results)
    
    print("🎉 Visualization dashboard demonstration complete!")

if __name__ == "__main__":
    demonstrate_dashboard() 