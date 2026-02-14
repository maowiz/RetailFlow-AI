# src/models/model_evaluator.py

"""
Model Evaluation Module

Provides comprehensive evaluation of forecasting models
with finance-relevant metrics and visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Comprehensive forecast evaluation with business-relevant metrics.
    
    Goes beyond standard ML metrics to include:
    - Financial impact metrics (revenue at risk, cost of error)
    - Directional accuracy (did we predict up/down correctly?)
    - Bias analysis (do we systematically over/under forecast?)
    - Segment-level performance (which stores/categories are worst?)
    - Visual diagnostics for stakeholder presentations
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[np.ndarray] = None,
        store_ids: Optional[np.ndarray] = None,
        categories: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ):
        """
        Initialize evaluator with actual and predicted values.
        
        Args:
            y_true: Actual sales values
            y_pred: Predicted sales values
            dates: Optional date array for time-based analysis
            store_ids: Optional store identifiers for segment analysis
            categories: Optional category labels for segment analysis
            model_name: Name for display purposes
        """
        self.y_true = np.asarray(y_true, dtype=np.float64)
        self.y_pred = np.asarray(y_pred, dtype=np.float64)
        self.dates = dates
        self.store_ids = store_ids
        self.categories = categories
        self.model_name = model_name
        
        # Precompute residuals
        self.residuals = self.y_true - self.y_pred
        self.abs_residuals = np.abs(self.residuals)
        self.pct_errors = np.where(
            self.y_true > 0,
            self.residuals / self.y_true,
            0
        )
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive set of evaluation metrics.
        
        Returns dictionary with all metrics.
        """
        metrics = {}
        
        # === Standard ML Metrics ===
        metrics['rmse'] = float(np.sqrt(
            mean_squared_error(self.y_true, self.y_pred)
        ))
        metrics['mae'] = float(mean_absolute_error(self.y_true, self.y_pred))
        metrics['r2'] = float(r2_score(self.y_true, self.y_pred))
        
        # MAPE (only on non-zero actuals)
        non_zero = self.y_true > 0
        if non_zero.sum() > 0:
            metrics['mape'] = float(mean_absolute_percentage_error(
                self.y_true[non_zero], self.y_pred[non_zero]
            ))
        
        # SMAPE
        denom = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2.0
        denom = np.where(denom == 0, 1.0, denom)
        metrics['smape'] = float(np.mean(np.abs(self.residuals) / denom))
        
        # WAPE
        total_actual = np.sum(np.abs(self.y_true))
        if total_actual > 0:
            metrics['wape'] = float(
                np.sum(self.abs_residuals) / total_actual
            )
        
        # === Bias Metrics ===
        metrics['mean_bias'] = float(np.mean(self.y_pred - self.y_true))
        metrics['median_bias'] = float(np.median(self.y_pred - self.y_true))
        metrics['bias_pct'] = float(
            metrics['mean_bias'] / np.mean(self.y_true) * 100
            if np.mean(self.y_true) > 0 else 0
        )
        
        # === Directional Accuracy ===
        # What percentage of time did we correctly predict
        # whether sales would go up or down vs previous period?
        if len(self.y_true) > 1:
            actual_direction = np.sign(np.diff(self.y_true))
            pred_direction = np.sign(np.diff(self.y_pred))
            metrics['directional_accuracy'] = float(
                np.mean(actual_direction == pred_direction)
            )
        
        # === Percentile Errors ===
        metrics['error_p50'] = float(np.percentile(self.abs_residuals, 50))
        metrics['error_p90'] = float(np.percentile(self.abs_residuals, 90))
        metrics['error_p95'] = float(np.percentile(self.abs_residuals, 95))
        metrics['error_p99'] = float(np.percentile(self.abs_residuals, 99))
        
        # === Financial Impact Metrics ===
        # Over-forecast cost (excess inventory holding)
        over_forecast = np.where(
            self.y_pred > self.y_true,
            self.y_pred - self.y_true,
            0
        )
        metrics['total_over_forecast'] = float(np.sum(over_forecast))
        metrics['avg_over_forecast'] = float(np.mean(over_forecast))
        
        # Under-forecast cost (potential stockouts)
        under_forecast = np.where(
            self.y_pred < self.y_true,
            self.y_true - self.y_pred,
            0
        )
        metrics['total_under_forecast'] = float(np.sum(under_forecast))
        metrics['avg_under_forecast'] = float(np.mean(under_forecast))
        
        # Forecast accuracy (1 - WAPE)
        metrics['forecast_accuracy'] = max(0.0, 1.0 - metrics.get('wape', 1.0))
        
        return metrics
    
    def segment_analysis(
        self,
        segment_col: str = 'store'
    ) -> pd.DataFrame:
        """
        Analyze forecast performance by segment (store or category).
        
        Identifies which segments have the best/worst forecast quality.
        Critical for operations teams to know where to focus attention.
        
        Args:
            segment_col: Which segmentation to use ('store' or 'category')
        
        Returns:
            DataFrame with per-segment metrics, sorted by RMSE
        """
        if segment_col == 'store' and self.store_ids is not None:
            segments = self.store_ids
        elif segment_col == 'category' and self.categories is not None:
            segments = self.categories
        else:
            logger.warning(f"No {segment_col} data available for segment analysis")
            return pd.DataFrame()
        
        results = []
        
        for segment in np.unique(segments):
            mask = segments == segment
            
            if mask.sum() < 5:
                continue
            
            y_t = self.y_true[mask]
            y_p = self.y_pred[mask]
            
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            mae = mean_absolute_error(y_t, y_p)
            
            # MAPE only on non-zero
            non_zero = y_t > 0
            mape = (
                mean_absolute_percentage_error(y_t[non_zero], y_p[non_zero])
                if non_zero.sum() > 0 else float('inf')
            )
            
            bias = np.mean(y_p - y_t)
            
            results.append({
                'segment': segment,
                'n_samples': int(mask.sum()),
                'avg_actual': float(np.mean(y_t)),
                'avg_predicted': float(np.mean(y_p)),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'bias': float(bias),
                'bias_pct': float(
                    bias / np.mean(y_t) * 100 if np.mean(y_t) > 0 else 0
                ),
            })
        
        df = pd.DataFrame(results).sort_values('rmse', ascending=True)
        
        logger.info(f"\nSegment Analysis ({segment_col}):")
        logger.info(f"  Best 3 segments (lowest RMSE):")
        for _, row in df.head(3).iterrows():
            logger.info(
                f"    {row['segment']}: RMSE={row['rmse']:.2f}, "
                f"MAPE={row['mape']:.4f}"
            )
        logger.info(f"  Worst 3 segments (highest RMSE):")
        for _, row in df.tail(3).iterrows():
            logger.info(
                f"    {row['segment']}: RMSE={row['rmse']:.2f}, "
                f"MAPE={row['mape']:.4f}"
            )
        
        return df
    
    def create_diagnostic_plots(self) -> Dict[str, go.Figure]:
        """
        Create comprehensive diagnostic visualizations.
        
        Returns dictionary of Plotly figures that can be 
        displayed in the Streamlit dashboard.
        """
        figures = {}
        
        # 1. Actual vs Predicted Scatter Plot
        figures['scatter'] = self._plot_actual_vs_predicted()
        
        # 2. Residual Distribution
        figures['residual_dist'] = self._plot_residual_distribution()
        
        # 3. Time Series Comparison (if dates available)
        if self.dates is not None:
            figures['time_series'] = self._plot_time_series_comparison()
        
        # 4. Error by Magnitude
        figures['error_by_magnitude'] = self._plot_error_by_magnitude()
        
        # 5. Forecast Bias Over Time (if dates available)
        if self.dates is not None:
            figures['bias_over_time'] = self._plot_bias_over_time()
        
        return figures
    
    def _plot_actual_vs_predicted(self) -> go.Figure:
        """Scatter plot of actual vs predicted with perfect prediction line."""
        
        # Sample if too many points (for performance)
        n = len(self.y_true)
        if n > 10000:
            idx = np.random.choice(n, 10000, replace=False)
            y_t = self.y_true[idx]
            y_p = self.y_pred[idx]
        else:
            y_t = self.y_true
            y_p = self.y_pred
        
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(go.Scattergl(
            x=y_t,
            y=y_p,
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(99, 110, 250, 0.3)',
                line=dict(width=0)
            ),
            name='Predictions',
            hovertemplate=(
                'Actual: %{x:.0f}<br>'
                'Predicted: %{y:.0f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Perfect prediction line
        max_val = max(np.max(y_t), np.max(y_p))
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'{self.model_name}: Actual vs Predicted Sales',
            xaxis_title='Actual Sales',
            yaxis_title='Predicted Sales',
            template='plotly_dark',
            width=700,
            height=500
        )
        
        return fig
    
    def _plot_residual_distribution(self) -> go.Figure:
        """Histogram of residuals — should be centered around 0."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.residuals,
            nbinsx=100,
            name='Residuals',
            marker_color='rgba(99, 110, 250, 0.7)',
            hovertemplate='Error: %{x:.0f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add vertical line at 0
        fig.add_vline(
            x=0, line_dash="dash", line_color="red",
            annotation_text="Zero Error"
        )
        
        # Add mean bias line
        mean_bias = np.mean(self.residuals)
        fig.add_vline(
            x=mean_bias, line_dash="dot", line_color="yellow",
            annotation_text=f"Mean Bias: {mean_bias:.1f}"
        )
        
        fig.update_layout(
            title=f'{self.model_name}: Residual Distribution',
            xaxis_title='Residual (Actual - Predicted)',
            yaxis_title='Count',
            template='plotly_dark',
            width=700,
            height=400
        )
        
        return fig
    
    def _plot_time_series_comparison(self) -> go.Figure:
        """
        Time series plot comparing actual vs predicted over time.
        Aggregated daily for readability.
        """
        df = pd.DataFrame({
            'date': self.dates,
            'actual': self.y_true,
            'predicted': self.y_pred
        })
        
        # Aggregate by date
        daily = df.groupby('date').agg({
            'actual': 'sum',
            'predicted': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['actual'],
            mode='lines',
            name='Actual Sales',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['predicted'],
            mode='lines',
            name='Predicted Sales',
            line=dict(color='#636EFA', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=f'{self.model_name}: Daily Sales — Actual vs Predicted',
            xaxis_title='Date',
            yaxis_title='Total Daily Sales',
            template='plotly_dark',
            width=900,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _plot_error_by_magnitude(self) -> go.Figure:
        """
        Show how error varies with sales magnitude.
        
        Important insight: Models typically have higher absolute 
        errors for high-volume items but lower percentage errors.
        """
        # Bin actuals into deciles
        df = pd.DataFrame({
            'actual': self.y_true,
            'abs_error': self.abs_residuals
        })
        
        # Remove zeros for meaningful binning
        df = df[df['actual'] > 0].copy()
        
        if len(df) < 100:
            return go.Figure()
        
        df['magnitude_bin'] = pd.qcut(
            df['actual'], q=10, duplicates='drop'
        )
        
        binned = df.groupby('magnitude_bin').agg({
            'actual': 'mean',
            'abs_error': ['mean', 'median', 'std']
        }).reset_index()
        
        binned.columns = [
            'bin', 'avg_actual', 'mean_error', 'median_error', 'std_error'
        ]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Absolute Error by Magnitude', 
                          'Relative Error by Magnitude']
        )
        
        fig.add_trace(go.Bar(
            x=[str(b) for b in binned['bin']],
            y=binned['mean_error'],
            name='Mean Abs Error',
            marker_color='#EF553B'
        ), row=1, col=1)
        
        # Relative error
        binned['rel_error'] = binned['mean_error'] / binned['avg_actual']
        
        fig.add_trace(go.Bar(
            x=[str(b) for b in binned['bin']],
            y=binned['rel_error'],
            name='Relative Error',
            marker_color='#FFA15A'
        ), row=1, col=2)
        
        fig.update_layout(
            title=f'{self.model_name}: Error Analysis by Sales Magnitude',
            template='plotly_dark',
            width=1000,
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _plot_bias_over_time(self) -> go.Figure:
        """
        Show forecast bias trends over time.
        
        If bias drifts, it indicates model degradation or
        concept drift — a signal to retrain.
        """
        df = pd.DataFrame({
            'date': self.dates,
            'residual': self.residuals
        })
        
        # Rolling 7-day average bias
        daily_bias = df.groupby('date')['residual'].mean().reset_index()
        daily_bias = daily_bias.sort_values('date')
        daily_bias['rolling_bias'] = (
            daily_bias['residual'].rolling(7, min_periods=1).mean()
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_bias['date'],
            y=daily_bias['residual'],
            mode='markers',
            marker=dict(size=3, color='rgba(99, 110, 250, 0.3)'),
            name='Daily Bias'
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_bias['date'],
            y=daily_bias['rolling_bias'],
            mode='lines',
            line=dict(color='red', width=2),
            name='7-day Rolling Avg Bias'
        ))
        
        # Zero line
        fig.add_hline(
            y=0, line_dash="dash", line_color="green",
            annotation_text="No Bias"
        )
        
        fig.update_layout(
            title=f'{self.model_name}: Forecast Bias Over Time',
            xaxis_title='Date',
            yaxis_title='Bias (Actual - Predicted)',
            template='plotly_dark',
            width=900,
            height=400
        )
        
        return fig
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a text-based evaluation report.
        
        This can be fed to Groq LLM for natural language insights.
        
        Returns:
            Formatted string report
        """
        metrics = self.calculate_all_metrics()
        
        report = f"""
{'='*60}
FORECAST EVALUATION REPORT: {self.model_name}
{'='*60}

📊 ACCURACY METRICS
  RMSE:                {metrics['rmse']:.2f}
  MAE:                 {metrics['mae']:.2f}
  MAPE:                {metrics.get('mape', 'N/A'):.4f} ({metrics.get('mape', 0)*100:.2f}%)
  SMAPE:               {metrics['smape']:.4f}
  WAPE:                {metrics.get('wape', 'N/A'):.4f}
  R²:                  {metrics['r2']:.4f}
  Forecast Accuracy:   {metrics['forecast_accuracy']*100:.1f}%

📈 BIAS ANALYSIS
  Mean Bias:           {metrics['mean_bias']:.2f}
  Median Bias:         {metrics['median_bias']:.2f}
  Bias %:              {metrics['bias_pct']:.2f}%
  Direction:           {'Over-forecasting ⬆️' if metrics['mean_bias'] > 0 else 'Under-forecasting ⬇️'}

📉 ERROR DISTRIBUTION
  50th percentile:     {metrics['error_p50']:.2f}
  90th percentile:     {metrics['error_p90']:.2f}
  95th percentile:     {metrics['error_p95']:.2f}
  99th percentile:     {metrics['error_p99']:.2f}

💰 FINANCIAL IMPACT
  Total Over-forecast: {metrics['total_over_forecast']:.0f} units
  Avg Over-forecast:   {metrics['avg_over_forecast']:.2f} units/prediction
  Total Under-forecast:{metrics['total_under_forecast']:.0f} units
  Avg Under-forecast:  {metrics['avg_under_forecast']:.2f} units/prediction

{'📐 DIRECTIONAL ACCURACY' if 'directional_accuracy' in metrics else ''}
{'  Accuracy: ' + f"{metrics['directional_accuracy']*100:.1f}%" if 'directional_accuracy' in metrics else ''}

{'='*60}
"""
        return report
