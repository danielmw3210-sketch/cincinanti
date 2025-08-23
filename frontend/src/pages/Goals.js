import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  Target, 
  Plus, 
  Filter, 
  Search, 
  Calendar,
  TrendingUp,
  Clock,
  CheckCircle,
  Edit,
  Trash2,
  Flag
} from 'lucide-react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmptyState from '../components/common/EmptyState';
import Modal from '../components/common/Modal';

const Goals = () => {
  const [filters, setFilters] = useState({
    status: '',
    category: '',
    priority: '',
    type: '',
    search: '',
  });
  const [selectedGoal, setSelectedGoal] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const { data: goals, isLoading, refetch } = useQuery(
    ['goals', filters],
    async () => {
      const params = new URLSearchParams();
      if (filters.status) params.append('status', filters.status);
      if (filters.category) params.append('category', filters.category);
      if (filters.priority) params.append('priority', filters.priority);
      if (filters.type) params.append('type', filters.type);
      if (filters.search) params.append('search', filters.search);
      
      const response = await api.get(`/api/goals?${params.toString()}`);
      return response.data.data;
    }
  );

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters({
      status: '',
      category: '',
      priority: '',
      type: '',
      search: '',
    });
  };

  const handleDelete = async () => {
    try {
      await api.delete(`/api/goals/${selectedGoal._id}`);
      toast.success('Goal deleted successfully');
      refetch();
      setIsDeleteModalOpen(false);
      setSelectedGoal(null);
    } catch (error) {
      toast.error('Failed to delete goal');
    }
  };

  const openDeleteModal = (goal) => {
    setSelectedGoal(goal);
    setIsDeleteModalOpen(true);
  };

  if (isLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const statusLabels = {
    'not-started': 'Not Started',
    'in-progress': 'In Progress',
    'completed': 'Completed',
    'on-hold': 'On Hold',
    'cancelled': 'Cancelled',
  };

  const priorityLabels = {
    'low': 'Low',
    'medium': 'Medium',
    'high': 'High',
    'urgent': 'Urgent',
  };

  const typeLabels = {
    'short-term': 'Short Term',
    'medium-term': 'Medium Term',
    'long-term': 'Long Term',
  };

  const getStatusColor = (status) => {
    const colors = {
      'not-started': 'text-gray-600 bg-gray-100',
      'in-progress': 'text-blue-600 bg-blue-100',
      'completed': 'text-green-600 bg-green-100',
      'on-hold': 'text-yellow-600 bg-yellow-100',
      'cancelled': 'text-red-600 bg-red-100',
    };
    return colors[status] || colors['not-started'];
  };

  const getPriorityColor = (priority) => {
    const colors = {
      'low': 'text-gray-600 bg-gray-100',
      'medium': 'text-blue-600 bg-blue-100',
      'high': 'text-orange-600 bg-orange-100',
      'urgent': 'text-red-600 bg-red-100',
    };
    return colors[priority] || colors['low'];
  };

  const getProgressColor = (progress) => {
    if (progress >= 80) return 'bg-green-500';
    if (progress >= 60) return 'bg-blue-500';
    if (progress >= 40) return 'bg-yellow-500';
    if (progress >= 20) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const isOverdue = (goal) => {
    if (!goal.targetDate || goal.status === 'completed') return false;
    return new Date(goal.targetDate) < new Date();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Goals</h1>
          <p className="text-gray-600">Track your personal and business goals, and measure your progress.</p>
        </div>
        <div className="flex space-x-3">
          <Link
            to="/goals/new"
            className="btn-primary inline-flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Goal
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search goals..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="form-input pl-10 w-full"
              />
            </div>
          </div>

          {/* Status Filter */}
          <div className="sm:w-48">
            <select
              value={filters.status}
              onChange={(e) => handleFilterChange('status', e.target.value)}
              className="form-input"
            >
              <option value="">All Statuses</option>
              {Object.entries(statusLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* Category Filter */}
          <div className="sm:w-48">
            <select
              value={filters.category}
              onChange={(e) => handleFilterChange('category', e.target.value)}
              className="form-input"
            >
              <option value="">All Categories</option>
              <option value="personal">Personal</option>
              <option value="business">Business</option>
              <option value="health">Health</option>
              <option value="career">Career</option>
              <option value="financial">Financial</option>
              <option value="learning">Learning</option>
            </select>
          </div>

          {/* Priority Filter */}
          <div className="sm:w-48">
            <select
              value={filters.priority}
              onChange={(e) => handleFilterChange('priority', e.target.value)}
              className="form-input"
            >
              <option value="">All Priorities</option>
              {Object.entries(priorityLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* Type Filter */}
          <div className="sm:w-48">
            <select
              value={filters.type}
              onChange={(e) => handleFilterChange('type', e.target.value)}
              className="form-input"
            >
              <option value="">All Types</option>
              {Object.entries(typeLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* Clear Filters */}
          <button
            onClick={clearFilters}
            className="btn-secondary whitespace-nowrap"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Goals List */}
      {goals?.length > 0 ? (
        <div className="space-y-4">
          {goals.map((goal) => (
            <div key={goal._id} className="card hover:shadow-md transition-shadow duration-200">
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="h-10 w-10 bg-green-100 rounded-full flex items-center justify-center">
                    <Target className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {goal.title}
                      </h3>
                      {isOverdue(goal) && (
                        <span className="badge-danger text-xs">
                          <Clock className="h-3 w-3 mr-1" />
                          Overdue
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-500 mt-1">
                      <Calendar className="h-4 w-4" />
                      <span>Due: {new Date(goal.targetDate).toLocaleDateString()}</span>
                      {goal.category && (
                        <>
                          <span>•</span>
                          <span className="capitalize">{goal.category}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {/* Status */}
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(goal.status)}`}>
                    {statusLabels[goal.status] || 'Unknown'}
                  </span>
                  
                  {/* Priority */}
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getPriorityColor(goal.priority)}`}>
                    {priorityLabels[goal.priority] || 'Unknown'}
                  </span>
                  
                  {/* Actions */}
                  <Link
                    to={`/goals/${goal._id}/edit`}
                    className="btn-secondary p-2"
                  >
                    <Edit className="h-4 w-4" />
                  </Link>
                  <button
                    onClick={() => openDeleteModal(goal)}
                    className="btn-danger p-2"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Description */}
              {goal.description && (
                <div className="mb-4">
                  <p className="text-gray-700">{goal.description}</p>
                </div>
              )}

              {/* Progress */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Progress</span>
                  <span className="text-sm text-gray-500">{goal.progress || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(goal.progress || 0)}`}
                    style={{ width: `${goal.progress || 0}%` }}
                  />
                </div>
              </div>

              {/* Meta Info */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm text-gray-600">
                <div className="flex items-center space-x-2">
                  <Flag className="h-4 w-4" />
                  <span>{typeLabels[goal.type] || 'Unknown'}</span>
                </div>
                {goal.startDate && (
                  <div className="flex items-center space-x-2">
                    <Calendar className="h-4 w-4" />
                    <span>Started: {new Date(goal.startDate).toLocaleDateString()}</span>
                  </div>
                )}
                {goal.milestones?.length > 0 && (
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4" />
                    <span>{goal.milestones.length} milestones</span>
                  </div>
                )}
                {goal.actions?.length > 0 && (
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4" />
                    <span>{goal.actions.length} actions</span>
                  </div>
                )}
              </div>

              {/* Tags */}
              {goal.tags?.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {goal.tags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={Target}
          title="No goals found"
          description="Start setting goals to track your progress and achievements."
          action={
            <Link to="/goals/new" className="btn-primary">
              Create Goal
            </Link>
          }
        />
      )}

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        title="Delete Goal"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-gray-600">
            Are you sure you want to delete "{selectedGoal?.title}"? 
            This action cannot be undone.
          </p>
          <div className="flex space-x-3 justify-end">
            <button
              onClick={() => setIsDeleteModalOpen(false)}
              className="btn-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              className="btn-danger"
            >
              Delete
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Goals;
