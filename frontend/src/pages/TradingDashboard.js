import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  BarChart3,
  Settings,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmptyState from '../components/common/EmptyState';

const TradingDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isTrading, setIsTrading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Simulate API calls - replace with actual API endpoints
  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      // This would be replaced with actual API calls
      const mockData = {
        portfolio: {
          totalBalance: 12547.32,
          availableBalance: 8234.56,
          unrealizedPnl: 234.78,
          totalReturn: 12.5,
          dailyPnl: 156.23
        },
        positions: [
          {
            pair: 'BTC/USD',
            size: 0.5,
            entryPrice: 45000,
            currentPrice: 46500,
            pnl: 750,
            pnlPercent: 3.33,
            type: 'long'
          },
          {
            pair: 'ETH/USD',
            size: 2.0,
            entryPrice: 3200,
            currentPrice: 3150,
            pnl: -100,
            pnlPercent: -1.56,
            type: 'long'
          }
        ],
        recentTrades: [
          {
            id: 1,
            pair: 'BTC/USD',
            type: 'buy',
            size: 0.1,
            price: 46000,
            timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
            pnl: 50,
            status: 'completed'
          },
          {
            id: 2,
            pair: 'ETH/USD',
            type: 'sell',
            size: 0.5,
            price: 3200,
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
            pnl: 25,
            status: 'completed'
          }
        ],
        performance: {
          totalTrades: 156,
          winningTrades: 89,
          losingTrades: 67,
          winRate: 57.1,
          profitFactor: 1.45,
          sharpeRatio: 1.23,
          maxDrawdown: -8.5
        },
        aiSignals: [
          {
            pair: 'BTC/USD',
            signal: 'buy',
            confidence: 0.78,
            timestamp: new Date(),
            reasoning: 'Strong bullish momentum with high volume'
          },
          {
            pair: 'ETH/USD',
            signal: 'hold',
            confidence: 0.45,
            timestamp: new Date(),
            reasoning: 'Mixed signals, waiting for clearer direction'
          }
        ],
        riskMetrics: {
          portfolioHeat: 0.15,
          maxDrawdown: 0.085,
          var95: 0.025,
          currentDrawdown: 0.02
        }
      };
      
      setDashboardData(mockData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const toggleTrading = () => {
    setIsTrading(!isTrading);
    // Here you would make an API call to start/stop trading
  };

  const refreshData = () => {
    fetchDashboardData();
  };

  if (isLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const stats = [
    {
      name: 'Total Balance',
      value: `$${dashboardData?.portfolio?.totalBalance?.toLocaleString() || '0'}`,
      change: `+${dashboardData?.portfolio?.totalReturn || 0}%`,
      changeType: 'positive',
      icon: DollarSign,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Available Balance',
      value: `$${dashboardData?.portfolio?.availableBalance?.toLocaleString() || '0'}`,
      change: 'Ready to trade',
      changeType: 'neutral',
      icon: Target,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Daily P&L',
      value: `$${dashboardData?.portfolio?.dailyPnl?.toFixed(2) || '0'}`,
      change: dashboardData?.portfolio?.dailyPnl >= 0 ? 'positive' : 'negative',
      changeType: dashboardData?.portfolio?.dailyPnl >= 0 ? 'positive' : 'negative',
      icon: TrendingUp,
      color: dashboardData?.portfolio?.dailyPnl >= 0 ? 'text-green-600' : 'text-red-600',
      bgColor: dashboardData?.portfolio?.dailyPnl >= 0 ? 'bg-green-100' : 'bg-red-100',
    },
    {
      name: 'Win Rate',
      value: `${dashboardData?.performance?.winRate || 0}%`,
      change: `${dashboardData?.performance?.totalTrades || 0} trades`,
      changeType: 'neutral',
      icon: BarChart3,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  ];

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'buy': return 'text-green-600 bg-green-100';
      case 'sell': return 'text-red-600 bg-red-100';
      case 'hold': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSignalIcon = (signal) => {
    switch (signal) {
      case 'buy': return TrendingUp;
      case 'sell': return TrendingDown;
      case 'hold': return Pause;
      default: return Activity;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Trading Dashboard</h1>
          <p className="text-gray-600">
            Real-time trading performance and AI signals
            {lastUpdate && (
              <span className="ml-2 text-sm text-gray-500">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={refreshData}
            className="btn-secondary inline-flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <button
            onClick={toggleTrading}
            className={`inline-flex items-center ${
              isTrading 
                ? 'btn-danger' 
                : 'btn-primary'
            }`}
          >
            {isTrading ? (
              <>
                <Pause className="h-4 w-4 mr-2" />
                Stop Trading
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Trading
              </>
            )}
          </button>
        </div>
      </div>

      {/* Trading Status */}
      <div className={`card ${isTrading ? 'border-green-200 bg-green-50' : 'border-yellow-200 bg-yellow-50'}`}>
        <div className="flex items-center">
          <div className={`flex-shrink-0 p-2 rounded-full ${isTrading ? 'bg-green-100' : 'bg-yellow-100'}`}>
            {isTrading ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
            )}
          </div>
          <div className="ml-3">
            <h3 className={`text-sm font-medium ${isTrading ? 'text-green-800' : 'text-yellow-800'}`}>
              {isTrading ? 'Trading Active' : 'Trading Paused'}
            </h3>
            <p className={`text-sm ${isTrading ? 'text-green-600' : 'text-yellow-600'}`}>
              {isTrading 
                ? 'AI is actively monitoring markets and executing trades'
                : 'Trading is paused. Click "Start Trading" to begin.'
              }
            </p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="card">
            <div className="flex items-center">
              <div className={`flex-shrink-0 p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                <p className={`text-sm ${
                  stat.changeType === 'positive' ? 'text-green-600' :
                  stat.changeType === 'negative' ? 'text-red-600' :
                  'text-gray-500'
                }`}>
                  {stat.change}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Current Positions */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Current Positions</h2>
            <span className="text-sm text-gray-500">
              {dashboardData?.positions?.length || 0} open
            </span>
          </div>
          <div className="space-y-4">
            {dashboardData?.positions?.length > 0 ? (
              dashboardData.positions.map((position, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex items-center space-x-3">
                    <div className={`flex-shrink-0 p-2 rounded-full ${
                      position.pnl >= 0 ? 'bg-green-100' : 'bg-red-100'
                    }`}>
                      {position.pnl >= 0 ? (
                        <TrendingUp className="h-4 w-4 text-green-600" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-600" />
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {position.pair}
                      </p>
                      <p className="text-sm text-gray-500">
                        {position.size} {position.type} @ ${position.entryPrice?.toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-medium ${
                      position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${position.pnl?.toFixed(2)} ({position.pnlPercent?.toFixed(2)}%)
                    </p>
                    <p className="text-sm text-gray-500">
                      Current: ${position.currentPrice?.toLocaleString()}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState
                icon={Target}
                title="No open positions"
                description="Start trading to see your positions here."
              />
            )}
          </div>
        </div>

        {/* AI Signals */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">AI Signals</h2>
            <span className="text-sm text-gray-500">
              Real-time analysis
            </span>
          </div>
          <div className="space-y-4">
            {dashboardData?.aiSignals?.length > 0 ? (
              dashboardData.aiSignals.map((signal, index) => {
                const SignalIcon = getSignalIcon(signal.signal);
                return (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`flex-shrink-0 p-2 rounded-full ${getSignalColor(signal.signal).split(' ')[1]}`}>
                        <SignalIcon className={`h-4 w-4 ${getSignalColor(signal.signal).split(' ')[0]}`} />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">
                          {signal.pair}
                        </p>
                        <p className="text-sm text-gray-500">
                          {signal.reasoning}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSignalColor(signal.signal)}`}>
                        {signal.signal.toUpperCase()}
                      </span>
                      <p className="text-sm text-gray-500 mt-1">
                        {(signal.confidence * 100).toFixed(0)}% confidence
                      </p>
                    </div>
                  </div>
                );
              })
            ) : (
              <EmptyState
                icon={Activity}
                title="No AI signals"
                description="AI is analyzing market conditions..."
              />
            )}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Performance Metrics</h2>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">
              {dashboardData?.performance?.totalTrades || 0}
            </p>
            <p className="text-sm text-gray-500">Total Trades</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {dashboardData?.performance?.winRate || 0}%
            </p>
            <p className="text-sm text-gray-500">Win Rate</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {dashboardData?.performance?.profitFactor || 0}
            </p>
            <p className="text-sm text-gray-500">Profit Factor</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">
              {dashboardData?.performance?.sharpeRatio || 0}
            </p>
            <p className="text-sm text-gray-500">Sharpe Ratio</p>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Risk Metrics</h2>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {(dashboardData?.riskMetrics?.portfolioHeat * 100 || 0).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Portfolio Heat</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">
              {(dashboardData?.riskMetrics?.maxDrawdown * 100 || 0).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Max Drawdown</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-yellow-600">
              {(dashboardData?.riskMetrics?.var95 * 100 || 0).toFixed(2)}%
            </p>
            <p className="text-sm text-gray-500">VaR (95%)</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-600">
              {(dashboardData?.riskMetrics?.currentDrawdown * 100 || 0).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Current Drawdown</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;