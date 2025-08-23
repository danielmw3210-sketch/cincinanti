const mongoose = require('mongoose');

const promptSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Prompt title is required'],
    trim: true,
    maxlength: [200, 'Title cannot be more than 200 characters']
  },
  content: {
    type: String,
    required: [true, 'Prompt content is required'],
    trim: true,
    maxlength: [2000, 'Content cannot be more than 2000 characters']
  },
  category: {
    type: String,
    required: [true, 'Prompt category is required'],
    enum: [
      'business',
      'personal',
      'health',
      'relationships',
      'finance',
      'spirituality',
      'creativity',
      'learning',
      'productivity',
      'mindfulness',
      'gratitude',
      'reflection',
      'planning',
      'challenge',
      'celebration'
    ]
  },
  subcategory: {
    type: String,
    trim: true,
    maxlength: [100, 'Subcategory cannot be more than 100 characters']
  },
  difficulty: {
    type: String,
    enum: ['easy', 'medium', 'hard', 'expert'],
    default: 'medium'
  },
  estimatedTime: {
    type: Number, // in minutes
    min: [1, 'Estimated time must be at least 1 minute'],
    max: [480, 'Estimated time cannot exceed 8 hours']
  },
  tags: [{
    type: String,
    trim: true,
    maxlength: [50, 'Tag cannot be more than 50 characters']
  }],
  promptType: {
    type: String,
    enum: ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'one-time', 'recurring'],
    default: 'daily'
  },
  frequency: {
    type: String,
    enum: ['once', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
    default: 'once'
  },
  businessRelevance: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical'],
    default: 'medium'
  },
  lifeArea: {
    type: [String],
    enum: [
      'career',
      'health',
      'relationships',
      'finances',
      'personal_growth',
      'spirituality',
      'recreation',
      'community',
      'family',
      'education'
    ]
  },
  expectedOutcome: {
    type: String,
    trim: true,
    maxlength: [500, 'Expected outcome cannot be more than 500 characters']
  },
  resources: [{
    title: String,
    url: String,
    type: {
      type: String,
      enum: ['article', 'video', 'book', 'podcast', 'tool', 'other']
    },
    description: String
  }],
  examples: [{
    title: String,
    description: String,
    outcome: String
  }],
  isActive: {
    type: Boolean,
    default: true
  },
  isFeatured: {
    type: Boolean,
    default: false
  },
  usageCount: {
    type: Number,
    default: 0
  },
  averageRating: {
    type: Number,
    default: 0,
    min: 0,
    max: 5
  },
  totalRatings: {
    type: Number,
    default: 0
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  lastUsed: Date,
  nextScheduled: Date
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for prompt difficulty score
promptSchema.virtual('difficultyScore').get(function() {
  const difficultyMap = { easy: 1, medium: 2, hard: 3, expert: 4 };
  return difficultyMap[this.difficulty] || 2;
});

// Virtual for business priority score
promptSchema.virtual('businessPriorityScore').get(function() {
  const priorityMap = { low: 1, medium: 2, high: 3, critical: 4 };
  return priorityMap[this.businessRelevance] || 2;
});

// Indexes for better query performance
promptSchema.index({ category: 1, isActive: 1 });
promptSchema.index({ promptType: 1, frequency: 1 });
promptSchema.index({ businessRelevance: 1, isActive: 1 });
promptSchema.index({ tags: 1 });
promptSchema.index({ isFeatured: 1, isActive: 1 });
promptSchema.index({ averageRating: -1, usageCount: -1 });

// Static method to find prompts by category
promptSchema.statics.findByCategory = function(category) {
  return this.find({ category, isActive: true }).sort({ averageRating: -1, usageCount: -1 });
};

// Static method to find featured prompts
promptSchema.statics.findFeatured = function() {
  return this.find({ isFeatured: true, isActive: true }).sort({ averageRating: -1 });
};

// Static method to find prompts by business relevance
promptSchema.statics.findByBusinessRelevance = function(relevance) {
  return this.find({ businessRelevance: relevance, isActive: true }).sort({ averageRating: -1 });
};

// Method to increment usage count
promptSchema.methods.incrementUsage = function() {
  this.usageCount += 1;
  this.lastUsed = new Date();
  return this.save();
};

// Method to add rating
promptSchema.methods.addRating = function(rating) {
  if (rating < 1 || rating > 5) {
    throw new Error('Rating must be between 1 and 5');
  }
  
  this.totalRatings += 1;
  this.averageRating = ((this.averageRating * (this.totalRatings - 1)) + rating) / this.totalRatings;
  return this.save();
};

module.exports = mongoose.model('Prompt', promptSchema);
