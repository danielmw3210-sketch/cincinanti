import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  Lightbulb, 
  Filter, 
  Search, 
  Star, 
  Clock, 
  Target,
  Plus,
  BookOpen
} from 'lucide-react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmptyState from '../components/common/EmptyState';

const Prompts = () => {
  const [filters, setFilters] = useState({
    category: '',
    difficulty: '',
    businessRelevance: '',
    search: '',
  });

  const { data: prompts, isLoading } = useQuery(
    ['prompts', filters],
    async () => {
      const params = new URLSearchParams();
      if (filters.category) params.append('category', filters.category);
      if (filters.difficulty) params.append('difficulty', filters.difficulty);
      if (filters.businessRelevance) params.append('businessRelevance', filters.businessRelevance);
      if (filters.search) params.append('search', filters.search);
      
      const response = await api.get(`/api/prompts?${params.toString()}`);
      return response.data.data;
    }
  );

  const { data: categories } = useQuery(
    'prompt-categories',
    async () => {
      const response = await api.get('/api/prompts');
      const allPrompts = response.data.data;
      const uniqueCategories = [...new Set(allPrompts.map(p => p.category))];
      return uniqueCategories.filter(Boolean);
    }
  );

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters({
      category: '',
      difficulty: '',
      businessRelevance: '',
      search: '',
    });
  };

  if (isLoading) {
    return <LoadingSpinner size="lg" className="py-12" />;
  }

  const difficultyLabels = {
    'beginner': 'Beginner',
    'intermediate': 'Intermediate',
    'advanced': 'Advanced',
  };

  const businessRelevanceLabels = {
    'high': 'High',
    'medium': 'Medium',
    'low': 'Low',
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Prompts</h1>
          <p className="text-gray-600">Discover prompts to guide your life reflection and growth.</p>
        </div>
        <div className="flex space-x-3">
          <Link
            to="/entries/new"
            className="btn-primary inline-flex items-center"
          >
            <BookOpen className="h-4 w-4 mr-2" />
            Use Prompt
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
                placeholder="Search prompts..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="form-input pl-10 w-full"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="sm:w-48">
            <select
              value={filters.category}
              onChange={(e) => handleFilterChange('category', e.target.value)}
              className="form-input"
            >
              <option value="">All Categories</option>
              {categories?.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>

          {/* Difficulty Filter */}
          <div className="sm:w-48">
            <select
              value={filters.difficulty}
              onChange={(e) => handleFilterChange('difficulty', e.target.value)}
              className="form-input"
            >
              <option value="">All Difficulties</option>
              {Object.entries(difficultyLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* Business Relevance Filter */}
          <div className="sm:w-48">
            <select
              value={filters.businessRelevance}
              onChange={(e) => handleFilterChange('businessRelevance', e.target.value)}
              className="form-input"
            >
              <option value="">All Relevance</option>
              {Object.entries(businessRelevanceLabels).map(([value, label]) => (
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

      {/* Prompts Grid */}
      {prompts?.length > 0 ? (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {prompts.map((prompt) => (
            <div key={prompt._id} className="card hover:shadow-lg transition-shadow duration-200">
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-8 w-8 bg-primary-100 rounded-lg flex items-center justify-center">
                    <Lightbulb className="h-4 w-4 text-primary-600" />
                  </div>
                  <span className="badge-primary">{prompt.category}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Star className="h-4 w-4 text-yellow-400 fill-current" />
                  <span className="text-sm text-gray-600">
                    {prompt.averageRating?.toFixed(1) || 'N/A'}
                  </span>
                </div>
              </div>

              {/* Content */}
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {prompt.title}
              </h3>
              <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                {prompt.content}
              </p>

              {/* Tags */}
              {prompt.tags?.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {prompt.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                    >
                      {tag}
                    </span>
                  ))}
                  {prompt.tags.length > 3 && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                      +{prompt.tags.length - 3}
                    </span>
                  )}
                </div>
              )}

              {/* Meta Info */}
              <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
                <div className="flex items-center space-x-4">
                  <span className="flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    {prompt.estimatedTime || '5-10 min'}
                  </span>
                  <span className="flex items-center">
                    <Target className="h-4 w-4 mr-1" />
                    {difficultyLabels[prompt.difficulty] || 'Unknown'}
                  </span>
                </div>
                <span className="text-xs">
                  Used {prompt.usageCount || 0} times
                </span>
              </div>

              {/* Actions */}
              <div className="flex space-x-2">
                <Link
                  to={`/entries/new?promptId=${prompt._id}`}
                  className="btn-primary flex-1 text-center"
                >
                  Use This Prompt
                </Link>
                <button className="btn-secondary p-2">
                  <Star className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={Lightbulb}
          title="No prompts found"
          description="Try adjusting your filters or search terms to find relevant prompts."
          action={
            <button onClick={clearFilters} className="btn-primary">
              Clear Filters
            </button>
          }
        />
      )}
    </div>
  );
};

export default Prompts;
