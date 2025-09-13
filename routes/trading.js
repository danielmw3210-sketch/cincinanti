const express = require('express');
const router = express.Router();
const { AITrader } = require('../ai_trader');
const { MarketAnalyzer } = require('../market_analyzer');
const { KrakenClient } = require('../kraken_client');
const { RiskManager } = require('../risk_manager');
const { TradingExecutor } = require('../trading_executor');
const { PerformanceAnalyzer } = require('../performance_analyzer');

// Initialize trading components
const krakenClient = new KrakenClient();
const marketAnalyzer = new MarketAnalyzer(krakenClient);
const aiTrader = new AITrader(marketAnalyzer);
const riskManager = new RiskManager();
const tradingExecutor = new TradingExecutor();
const performanceAnalyzer = new PerformanceAnalyzer();

// Get trading dashboard data
router.get('/dashboard', async (req, res) => {
  try {
    // Get portfolio summary
    const portfolioSummary = tradingExecutor.get_portfolio_summary();
    
    // Get AI signals for major pairs
    const pairs = ['BTC/USD', 'ETH/USD', 'ADA/USD'];
    const aiSignals = [];
    
    for (const pair of pairs) {
      try {
        const signal = aiTrader.predict_signal(pair);
        if (signal && signal.signal !== 'hold') {
          aiSignals.push({
            pair,
            signal: signal.signal,
            confidence: signal.confidence,
            timestamp: new Date(),
            reasoning: signal.reason || 'AI analysis'
          });
        }
      } catch (error) {
        console.error(`Error getting signal for ${pair}:`, error);
      }
    }
    
    // Get performance metrics
    const performanceMetrics = performanceAnalyzer.calculate_comprehensive_metrics();
    
    // Get risk metrics
    const riskMetrics = riskManager.get_risk_metrics();
    
    // Get recent trades (mock data for now)
    const recentTrades = [
      {
        id: 1,
        pair: 'BTC/USD',
        type: 'buy',
        size: 0.1,
        price: 45000,
        timestamp: new Date(Date.now() - 1000 * 60 * 30),
        pnl: 50,
        status: 'completed'
      }
    ];
    
    const dashboardData = {
      portfolio: {
        totalBalance: portfolioSummary.portfolio?.total_balance || 10000,
        availableBalance: portfolioSummary.portfolio?.available_balance || 8000,
        unrealizedPnl: portfolioSummary.unrealized_pnl || 0,
        totalReturn: performanceMetrics.total_return_pct || 0,
        dailyPnl: riskMetrics.daily_pnl || 0
      },
      positions: Object.entries(portfolioSummary.positions || {}).map(([pair, pos]) => ({
        pair,
        size: pos.size || 0,
        entryPrice: pos.avg_price || 0,
        currentPrice: 45000, // This would be fetched from market data
        pnl: pos.unrealized_pnl || 0,
        pnlPercent: pos.unrealized_pnl ? (pos.unrealized_pnl / (pos.avg_price * pos.size)) * 100 : 0,
        type: 'long'
      })),
      recentTrades,
      performance: {
        totalTrades: performanceMetrics.total_trades || 0,
        winningTrades: performanceMetrics.winning_trades || 0,
        losingTrades: performanceMetrics.losing_trades || 0,
        winRate: performanceMetrics.win_rate_pct || 0,
        profitFactor: performanceMetrics.profit_factor || 0,
        sharpeRatio: performanceMetrics.sharpe_ratio || 0,
        maxDrawdown: performanceMetrics.max_drawdown_pct || 0
      },
      aiSignals,
      riskMetrics: {
        portfolioHeat: riskMetrics.portfolio_heat || 0,
        maxDrawdown: riskMetrics.max_drawdown || 0,
        var95: performanceMetrics.var_95 || 0,
        currentDrawdown: riskMetrics.current_drawdown || 0
      }
    };
    
    res.json({
      success: true,
      data: dashboardData
    });
    
  } catch (error) {
    console.error('Error getting trading dashboard data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get trading dashboard data',
      message: error.message
    });
  }
});

// Start/stop trading
router.post('/start', async (req, res) => {
  try {
    const { pair = 'BTC/USD' } = req.body;
    
    const result = tradingExecutor.start_trading_session(pair);
    
    res.json({
      success: result.success,
      data: result,
      message: result.success ? 'Trading started successfully' : result.reason
    });
    
  } catch (error) {
    console.error('Error starting trading:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to start trading',
      message: error.message
    });
  }
});

router.post('/stop', async (req, res) => {
  try {
    tradingExecutor.stop_trading();
    
    res.json({
      success: true,
      message: 'Trading stopped successfully'
    });
    
  } catch (error) {
    console.error('Error stopping trading:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to stop trading',
      message: error.message
    });
  }
});

// Get AI signal for a specific pair
router.get('/signal/:pair', async (req, res) => {
  try {
    const { pair } = req.params;
    
    const signal = aiTrader.predict_signal(pair);
    
    res.json({
      success: true,
      data: signal
    });
    
  } catch (error) {
    console.error(`Error getting signal for ${req.params.pair}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to get AI signal',
      message: error.message
    });
  }
});

// Get performance metrics
router.get('/performance', async (req, res) => {
  try {
    const metrics = performanceAnalyzer.calculate_comprehensive_metrics();
    
    res.json({
      success: true,
      data: metrics
    });
    
  } catch (error) {
    console.error('Error getting performance metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get performance metrics',
      message: error.message
    });
  }
});

// Get risk metrics
router.get('/risk', async (req, res) => {
  try {
    const metrics = riskManager.get_risk_metrics();
    
    res.json({
      success: true,
      data: metrics
    });
    
  } catch (error) {
    console.error('Error getting risk metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get risk metrics',
      message: error.message
    });
  }
});

// Train models
router.post('/train', async (req, res) => {
  try {
    const { pair = 'BTC/USD' } = req.body;
    
    const accuracies = aiTrader.train_models(pair);
    
    res.json({
      success: true,
      data: accuracies,
      message: 'Models trained successfully'
    });
    
  } catch (error) {
    console.error('Error training models:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to train models',
      message: error.message
    });
  }
});

module.exports = router;