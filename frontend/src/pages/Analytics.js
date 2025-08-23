import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  BarChart3, 
  TrendingUp, 
  Calendar,
  Target,
  Heart,
  Zap,
  Filter
} from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import api from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

const Analytics = () => {
  const [dateRange, setDateRange] = useState('month');

  const { data: overview, isLoading: overviewLoading } = useQuery(
    ['analytics-overview', dateRange],
    async () => {
      const response = await api.get(`/api/analytics/overview?dateRange=${dateRange}`);
      return response.data.data;
    }
  );

  const { data: moodAnalytics, isLoading: moodLoading } = useQuery(
    ['analytics-mood', dateRange],
    async () => {
      const response = await api.get(`/api/analytics/mood?dateRange=${dateRange}`);
      return response.data.data;
    }
  );

  const { data: businessAnalytics, isLoading: businessLoading } = useQuery(
    ['analytics-business', dateRange],
    async () => {
      const response = await api.get(`/api/analytics/business?dateRange=${dateRange}`);
      return response.data.data;
    }
  );

  if (overviewLoading || moodLoading || businessLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const moodColors = ['#10B981', '#3B82F6', '#6B7280', '#F59E0B', '#EF4444'];
  const chartColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const getMoodLabel = (mood) => {
    const labels = {
      'excellent': 'Excellent',
      'good': 'Good',
      'neutral': 'Neutral',
      'poor': 'Poor',
      'terrible': 'Terrible',
    };
    return labels[mood] || mood;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600">Insights and trends from your life tracking data.</p>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="form-input"
          >
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="quarter">This Quarter</option>
            <option value="year">This Year</option>
          </select>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-blue-100">
              <BarChart3 className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Entries</p>
              <p className="text-2xl font-semibold text-gray-900">
                {overview?.totalEntries || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-green-100">
              <Target className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active Goals</p>
              <p className="text-2xl font-semibold text-gray-900">
                {overview?.activeGoals || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-purple-100">
              <TrendingUp className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg. Mood</p>
              <p className="text-2xl font-semibold text-gray-900">
                {overview?.averageMood?.toFixed(1) || 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-orange-100">
              <Zap className="h-6 w-6 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg. Energy</p>
              <p className="text-2xl font-semibold text-gray-900">
                {overview?.averageEnergy?.toFixed(1) || 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Mood Trend */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Mood Trend</h2>
            <p className="card-subtitle">Your mood over time</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={moodAnalytics?.trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  domain={[1, 5]}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  labelFormatter={formatDate}
                  formatter={(value) => [getMoodLabel(value), 'Mood']}
                />
                <Line 
                  type="monotone" 
                  dataKey="mood" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Mood Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Mood Distribution</h2>
            <p className="card-subtitle">Breakdown of your moods</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={moodAnalytics?.distribution || []}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {moodAnalytics?.distribution?.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={moodColors[index % moodColors.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Energy vs Productivity */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Energy vs Productivity</h2>
            <p className="card-subtitle">Correlation between energy and productivity</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={overview?.energyProductivityData || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  domain={[0, 10]}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  labelFormatter={formatDate}
                />
                <Line 
                  type="monotone" 
                  dataKey="energy" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  name="Energy"
                />
                <Line 
                  type="monotone" 
                  dataKey="productivity" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  name="Productivity"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Business Metrics */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Business Metrics</h2>
            <p className="card-subtitle">Key business performance indicators</p>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={businessAnalytics?.metrics || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="metric" 
                  tick={{ fontSize: 12 }}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#8B5CF6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Key Insights</h2>
          <p className="card-subtitle">What your data tells us</p>
        </div>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {overview?.insights?.map((insight, index) => (
            <div key={index} className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <div className={`w-3 h-3 rounded-full bg-${insight.type === 'positive' ? 'green' : insight.type === 'warning' ? 'yellow' : 'blue'}-500`} />
                <span className="text-sm font-medium text-gray-700 capitalize">
                  {insight.type}
                </span>
              </div>
              <p className="text-sm text-gray-600">{insight.message}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      {overview?.recommendations?.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Recommendations</h2>
            <p className="card-subtitle">Actions you can take based on your data</p>
          </div>
          <div className="space-y-4">
            {overview.recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                <div className="flex-shrink-0 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">{index + 1}</span>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                    {recommendation.title}
                  </h4>
                  <p className="text-sm text-gray-600">
                    {recommendation.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;
