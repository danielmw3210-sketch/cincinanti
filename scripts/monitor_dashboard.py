"""Simple monitoring dashboard for the AI Crypto Trader."""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any
import requests
from flask import Flask, render_template_string, jsonify
from loguru import logger

from main import AICryptoTrader

app = Flask(__name__)

# Global trader instance
trader = None
dashboard_data = {}

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Crypto Trader Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .metric-label { color: #666; font-size: 0.9em; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #3498db; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background-color: #27ae60; }
        .status-stopped { background-color: #e74c3c; }
        .status-warning { background-color: #f39c12; }
        .trades-table { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Crypto Trader Dashboard</h1>
            <p>Real-time monitoring and control</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">System Status</div>
                <div class="metric-value">
                    <span class="status-indicator" id="system-status"></span>
                    <span id="system-text">Loading...</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value" id="portfolio-value">$0.00</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value" id="total-return">0.00%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Daily P&L</div>
                <div class="metric-value" id="daily-pnl">$0.00</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value" id="open-positions">0</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value" id="total-trades">0</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" id="max-drawdown">0.00%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value" id="win-rate">0.00%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Portfolio Performance</h3>
            <canvas id="portfolioChart" width="400" height="200"></canvas>
        </div>
        
        <div class="trades-table">
            <h3 style="padding: 20px 20px 0;">Recent Trades</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Pair</th>
                        <th>Action</th>
                        <th>Size</th>
                        <th>Price</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody id="trades-table-body">
                </tbody>
            </table>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
        </div>
    </div>

    <script>
        let portfolioChart;
        
        function initChart() {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        function updateDashboard(data) {
            // Update system status
            const systemStatus = document.getElementById('system-status');
            const systemText = document.getElementById('system-text');
            
            if (data.system_running && data.trading_active) {
                systemStatus.className = 'status-indicator status-running';
                systemText.textContent = 'Running';
            } else if (data.system_running) {
                systemStatus.className = 'status-indicator status-warning';
                systemText.textContent = 'Paused';
            } else {
                systemStatus.className = 'status-indicator status-stopped';
                systemText.textContent = 'Stopped';
            }
            
            // Update metrics
            document.getElementById('portfolio-value').textContent = 
                '$' + (data.portfolio?.total_balance || 0).toFixed(2);
            
            const totalReturn = data.performance?.total_return || 0;
            const totalReturnElement = document.getElementById('total-return');
            totalReturnElement.textContent = (totalReturn * 100).toFixed(2) + '%';
            totalReturnElement.className = totalReturn >= 0 ? 'metric-value positive' : 'metric-value negative';
            
            const dailyPnl = data.performance?.daily_pnl || 0;
            const dailyPnlElement = document.getElementById('daily-pnl');
            dailyPnlElement.textContent = '$' + dailyPnl.toFixed(2);
            dailyPnlElement.className = dailyPnl >= 0 ? 'metric-value positive' : 'metric-value negative';
            
            document.getElementById('open-positions').textContent = 
                Object.keys(data.portfolio?.positions || {}).length;
            
            document.getElementById('total-trades').textContent = 
                data.portfolio?.total_trades || 0;
            
            document.getElementById('max-drawdown').textContent = 
                ((data.risk_metrics?.max_drawdown || 0) * 100).toFixed(2) + '%';
            
            // Update chart
            if (data.portfolio_history && portfolioChart) {
                const labels = data.portfolio_history.map(h => 
                    new Date(h.timestamp).toLocaleTimeString()
                );
                const values = data.portfolio_history.map(h => h.portfolio_value);
                
                portfolioChart.data.labels = labels;
                portfolioChart.data.datasets[0].data = values;
                portfolioChart.update();
            }
            
            // Update trades table
            updateTradesTable(data.recent_trades || []);
        }
        
        function updateTradesTable(trades) {
            const tbody = document.getElementById('trades-table-body');
            tbody.innerHTML = '';
            
            trades.slice(-10).reverse().forEach(trade => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${new Date(trade.timestamp).toLocaleString()}</td>
                    <td>${trade.pair}</td>
                    <td><span class="${trade.signal === 'buy' ? 'positive' : 'negative'}">${trade.signal.toUpperCase()}</span></td>
                    <td>${trade.size}</td>
                    <td>$${trade.price.toFixed(2)}</td>
                    <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">$${(trade.pnl || 0).toFixed(2)}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error:', error));
        }
        
        // Initialize chart and refresh data
        initChart();
        refreshData();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        if trader:
            status = trader.get_status()
            return jsonify(status)
        else:
            return jsonify({'error': 'Trader not initialized'})
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/start', methods=['POST'])
def api_start():
    """API endpoint to start trading."""
    try:
        if trader:
            trader.start()
            return jsonify({'success': True, 'message': 'Trading started'})
        else:
            return jsonify({'error': 'Trader not initialized'})
    except Exception as e:
        logger.error(f"Error starting trader: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """API endpoint to stop trading."""
    try:
        if trader:
            trader.stop()
            return jsonify({'success': True, 'message': 'Trading stopped'})
        else:
            return jsonify({'error': 'Trader not initialized'})
    except Exception as e:
        logger.error(f"Error stopping trader: {e}")
        return jsonify({'error': str(e)})

def start_dashboard(trader_instance=None):
    """Start the monitoring dashboard."""
    global trader
    trader = trader_instance
    
    logger.info("Starting monitoring dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    # Initialize trader
    trader = AICryptoTrader()
    
    # Start dashboard
    start_dashboard(trader)