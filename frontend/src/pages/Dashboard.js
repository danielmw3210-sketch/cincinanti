import React from 'react';
import { useQuery } from 'react-query';
import { 
  TrendingUp, 
  Target, 
  BookOpen, 
  Lightbulb,
  Plus,
  Calendar,
  BarChart3
} from 'lucide-react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmptyState from '../components/common/EmptyState';

const Dashboard = () => {
  const { data: overview, isLoading } = useQuery(
    'dashboard-overview',
    async () => {
      const response = await api.get('/api/analytics/overview');
      return response.data.data;
    },
    {
      refetchInterval: 300000, // Refetch every 5 minutes
    }
  );

  const { data: recentEntries } = useQuery(
    'recent-entries',
    async () => {
      const response = await api.get('/api/life-entries?limit=5&sort=-createdAt');
      return response.data.data;
    }
  );

  const { data: upcomingGoals } = useQuery(
    'upcoming-goals',
    async () => {
      const response = await api.get('/api/goals?status=in-progress&sort=targetDate&limit=5');
      return response.data.data;
    }
  );

  if (isLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const stats = [
    {
      name: 'Total Entries',
      value: overview?.totalEntries || 0,
      icon: BookOpen,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Active Goals',
      value: overview?.activeGoals || 0,
      icon: Target,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Completed Goals',
      value: overview?.completedGoals || 0,
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Available Prompts',
      value: overview?.availablePrompts || 0,
      icon: Lightbulb,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ];

  const quickActions = [
    {
      name: 'Add Life Entry',
      description: 'Record your daily thoughts and experiences',
      href: '/entries/new',
      icon: BookOpen,
      color: 'bg-blue-500',
    },
    {
      name: 'Set New Goal',
      description: 'Create a new goal to work towards',
      href: '/goals/new',
      icon: Target,
      color: 'bg-green-500',
    },
    {
      name: 'Browse Prompts',
      description: 'Find prompts to guide your reflection',
      href: '/prompts',
      icon: Lightbulb,
      color: 'bg-orange-500',
    },
    {
      name: 'View Analytics',
      description: 'See your progress and insights',
      href: '/analytics',
      icon: BarChart3,
      color: 'bg-purple-500',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">Welcome back! Here's an overview of your life tracking journey.</p>
        </div>
        <div className="flex space-x-3">
          <Link
            to="/entries/new"
            className="btn-primary inline-flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Entry
          </Link>
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
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Quick Actions</h2>
          <p className="card-subtitle">Get started with these common tasks</p>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {quickActions.map((action) => (
            <Link
              key={action.name}
              to={action.href}
              className="group relative rounded-lg border border-gray-200 bg-white p-6 hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex items-center">
                <div className={`flex-shrink-0 p-3 rounded-lg ${action.color}`}>
                  <action.icon className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4">
                  <h3 className="text-sm font-medium text-gray-900 group-hover:text-primary-600">
                    {action.name}
                  </h3>
                  <p className="text-sm text-gray-500">{action.description}</p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Entries */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Recent Life Entries</h2>
            <Link
              to="/entries"
              className="text-sm text-primary-600 hover:text-primary-500"
            >
              View all
            </Link>
          </div>
          <div className="space-y-4">
            {recentEntries?.length > 0 ? (
              recentEntries.map((entry) => (
                <div
                  key={entry._id}
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex-shrink-0">
                    <div className="h-8 w-8 bg-primary-100 rounded-full flex items-center justify-center">
                      <BookOpen className="h-4 w-4 text-primary-600" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {entry.title || 'Untitled Entry'}
                    </p>
                    <p className="text-sm text-gray-500">
                      {new Date(entry.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex-shrink-0">
                    <span className={`badge-${getMoodBadgeColor(entry.mood)}`}>
                      {entry.mood}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState
                icon={BookOpen}
                title="No entries yet"
                description="Start tracking your life by creating your first entry."
                action={
                  <Link to="/entries/new" className="btn-primary">
                    Create Entry
                  </Link>
                }
              />
            )}
          </div>
        </div>

        {/* Upcoming Goals */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Upcoming Goals</h2>
            <Link
              to="/goals"
              className="text-sm text-primary-600 hover:text-primary-500"
            >
              View all
            </Link>
          </div>
          <div className="space-y-4">
            {upcomingGoals?.length > 0 ? (
              upcomingGoals.map((goal) => (
                <div
                  key={goal._id}
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex-shrink-0">
                    <div className="h-8 w-8 bg-green-100 rounded-full flex items-center justify-center">
                      <Target className="h-4 w-4 text-green-600" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {goal.title}
                    </p>
                    <p className="text-sm text-gray-500">
                      Due: {new Date(goal.targetDate).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex-shrink-0">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-600 h-2 rounded-full"
                          style={{ width: `${goal.progress || 0}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">
                        {goal.progress || 0}%
                      </span>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState
                icon={Target}
                title="No active goals"
                description="Set some goals to start tracking your progress."
                action={
                  <Link to="/goals/new" className="btn-primary">
                    Create Goal
                  </Link>
                }
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to get mood badge color
const getMoodBadgeColor = (mood) => {
  const moodColors = {
    'excellent': 'success',
    'good': 'success',
    'neutral': 'secondary',
    'poor': 'warning',
    'terrible': 'danger',
  };
  return moodColors[mood?.toLowerCase()] || 'secondary';
};

export default Dashboard;
