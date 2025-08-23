import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  BookOpen, 
  Plus, 
  Filter, 
  Search, 
  Calendar,
  Heart,
  TrendingUp,
  Target,
  Edit,
  Trash2
} from 'lucide-react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmptyState from '../components/common/EmptyState';
import Modal from '../components/common/Modal';

const LifeEntries = () => {
  const [filters, setFilters] = useState({
    mood: '',
    category: '',
    dateRange: '',
    search: '',
  });
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const { data: entries, isLoading, refetch } = useQuery(
    ['life-entries', filters],
    async () => {
      const params = new URLSearchParams();
      if (filters.mood) params.append('mood', filters.mood);
      if (filters.category) params.append('category', filters.category);
      if (filters.dateRange) params.append('dateRange', filters.dateRange);
      if (filters.search) params.append('search', filters.search);
      
      const response = await api.get(`/api/life-entries?${params.toString()}`);
      return response.data.data;
    }
  );

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters({
      mood: '',
      category: '',
      dateRange: '',
      search: '',
    });
  };

  const handleDelete = async () => {
    try {
      await api.delete(`/api/life-entries/${selectedEntry._id}`);
      toast.success('Entry deleted successfully');
      refetch();
      setIsDeleteModalOpen(false);
      setSelectedEntry(null);
    } catch (error) {
      toast.error('Failed to delete entry');
    }
  };

  const openDeleteModal = (entry) => {
    setSelectedEntry(entry);
    setIsDeleteModalOpen(true);
  };

  if (isLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const moodLabels = {
    'excellent': 'Excellent',
    'good': 'Good',
    'neutral': 'Neutral',
    'poor': 'Poor',
    'terrible': 'Terrible',
  };

  const getMoodColor = (mood) => {
    const colors = {
      'excellent': 'text-green-600 bg-green-100',
      'good': 'text-blue-600 bg-blue-100',
      'neutral': 'text-gray-600 bg-gray-100',
      'poor': 'text-yellow-600 bg-yellow-100',
      'terrible': 'text-red-600 bg-red-100',
    };
    return colors[mood?.toLowerCase()] || colors.neutral;
  };

  const getMoodIcon = (mood) => {
    const icons = {
      'excellent': '😊',
      'good': '🙂',
      'neutral': '😐',
      'poor': '😕',
      'terrible': '😢',
    };
    return icons[mood?.toLowerCase()] || icons.neutral;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Life Entries</h1>
          <p className="text-gray-600">Track your daily experiences, thoughts, and insights.</p>
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

      {/* Filters */}
      <div className="card">
        <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search entries..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="form-input pl-10 w-full"
              />
            </div>
          </div>

          {/* Mood Filter */}
          <div className="sm:w-48">
            <select
              value={filters.mood}
              onChange={(e) => handleFilterChange('mood', e.target.value)}
              className="form-input"
            >
              <option value="">All Moods</option>
              {Object.entries(moodLabels).map(([value, label]) => (
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
              <option value="relationships">Relationships</option>
              <option value="learning">Learning</option>
            </select>
          </div>

          {/* Date Range Filter */}
          <div className="sm:w-48">
            <select
              value={filters.dateRange}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
              className="form-input"
            >
              <option value="">All Time</option>
              <option value="today">Today</option>
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="quarter">This Quarter</option>
              <option value="year">This Year</option>
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

      {/* Entries List */}
      {entries?.length > 0 ? (
        <div className="space-y-4">
          {entries.map((entry) => (
            <div key={entry._id} className="card hover:shadow-md transition-shadow duration-200">
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="h-10 w-10 bg-primary-100 rounded-full flex items-center justify-center">
                    <BookOpen className="h-5 w-5 text-primary-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {entry.title || 'Untitled Entry'}
                    </h3>
                    <div className="flex items-center space-x-2 text-sm text-gray-500">
                      <Calendar className="h-4 w-4" />
                      <span>{new Date(entry.createdAt).toLocaleDateString()}</span>
                      {entry.category && (
                        <>
                          <span>•</span>
                          <span className="capitalize">{entry.category}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {/* Mood */}
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${getMoodColor(entry.mood)}`}>
                    <span className="mr-1">{getMoodIcon(entry.mood)}</span>
                    {moodLabels[entry.mood?.toLowerCase()] || 'Unknown'}
                  </div>
                  
                  {/* Actions */}
                  <Link
                    to={`/entries/${entry._id}/edit`}
                    className="btn-secondary p-2"
                  >
                    <Edit className="h-4 w-4" />
                  </Link>
                  <button
                    onClick={() => openDeleteModal(entry)}
                    className="btn-danger p-2"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Content */}
              {entry.content && (
                <div className="mb-4">
                  <p className="text-gray-700 line-clamp-3">{entry.content}</p>
                </div>
              )}

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {entry.energy && (
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600">
                      Energy: {entry.energy}/10
                    </span>
                  </div>
                )}
                {entry.productivity && (
                  <div className="flex items-center space-x-2">
                    <Target className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">
                      Productivity: {entry.productivity}/10
                    </span>
                  </div>
                )}
                {entry.stress && (
                  <div className="flex items-center space-x-2">
                    <Heart className="h-4 w-4 text-red-500" />
                    <span className="text-sm text-gray-600">
                      Stress: {entry.stress}/10
                    </span>
                  </div>
                )}
                {entry.satisfaction && (
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                    <span className="text-sm text-gray-600">
                      Satisfaction: {entry.satisfaction}/10
                    </span>
                  </div>
                )}
              </div>

              {/* Tags */}
              {entry.tags?.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {entry.tags.map((tag) => (
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
          icon={BookOpen}
          title="No entries found"
          description="Start tracking your life by creating your first entry."
          action={
            <Link to="/entries/new" className="btn-primary">
              Create Entry
            </Link>
          }
        />
      )}

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        title="Delete Entry"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-gray-600">
            Are you sure you want to delete "{selectedEntry?.title || 'this entry'}"? 
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

export default LifeEntries;
