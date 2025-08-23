const mongoose = require('mongoose');

const lifeEntrySchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  prompt: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Prompt',
    required: true
  },
  title: {
    type: String,
    required: [true, 'Entry title is required'],
    trim: true,
    maxlength: [200, 'Title cannot be more than 200 characters']
  },
  content: {
    type: String,
    required: [true, 'Entry content is required'],
    trim: true,
    maxlength: [10000, 'Content cannot be more than 10000 characters']
  },
  mood: {
    type: String,
    enum: ['terrible', 'bad', 'okay', 'good', 'great', 'excellent'],
    required: true
  },
  energy: {
    type: Number,
    min: 1,
    max: 10,
    required: true
  },
  productivity: {
    type: Number,
    min: 1,
    max: 10
  },
  stress: {
    type: Number,
    min: 1,
    max: 10
  },
  satisfaction: {
    type: Number,
    min: 1,
    max: 10
  },
  insights: [{
    type: String,
    trim: true,
    maxlength: [500, 'Insight cannot be more than 500 characters']
  }],
  actions: [{
    title: String,
    description: String,
    completed: { type: Boolean, default: false },
    dueDate: Date,
    priority: {
      type: String,
      enum: ['low', 'medium', 'high', 'urgent'],
      default: 'medium'
    }
  }],
  gratitude: [{
    type: String,
    trim: true,
    maxlength: [200, 'Gratitude item cannot be more than 200 characters']
  }],
  challenges: [{
    description: String,
    impact: {
      type: String,
      enum: ['low', 'medium', 'high', 'critical'],
      default: 'medium'
    },
    status: {
      type: String,
      enum: ['active', 'resolved', 'mitigated'],
      default: 'active'
    },
    lessons: String
  }],
  wins: [{
    title: String,
    description: String,
    impact: {
      type: String,
      enum: ['small', 'medium', 'large', 'huge'],
      default: 'medium'
    },
    celebration: String
  }],
  businessMetrics: {
    revenue: Number,
    customers: Number,
    projects: Number,
    meetings: Number,
    decisions: Number,
    innovations: Number
  },
  personalMetrics: {
    exercise: Number, // minutes
    sleep: Number, // hours
    reading: Number, // minutes
    meditation: Number, // minutes
    social: Number, // minutes
    learning: Number // minutes
  },
  tags: [{
    type: String,
    trim: true,
    maxlength: [50, 'Tag cannot be more than 50 characters']
  }],
  isPrivate: {
    type: Boolean,
    default: false
  },
  isCompleted: {
    type: Boolean,
    default: true
  },
  completionTime: Number, // in minutes
  rating: {
    type: Number,
    min: 1,
    max: 5
  },
  reflection: {
    whatWentWell: String,
    whatCouldBeBetter: String,
    whatILearned: String,
    whatToDoNext: String
  },
  attachments: [{
    filename: String,
    originalName: String,
    mimeType: String,
    size: Number,
    url: String
  }],
  location: {
    type: {
      type: String,
      enum: ['Point'],
      default: 'Point'
    },
    coordinates: [Number] // [longitude, latitude]
  },
  weather: {
    temperature: Number,
    condition: String,
    humidity: Number
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for entry summary
lifeEntrySchema.virtual('summary').get(function() {
  return this.content.length > 150 
    ? this.content.substring(0, 150) + '...' 
    : this.content;
});

// Virtual for overall score
lifeEntrySchema.virtual('overallScore').get(function() {
  const scores = [
    this.moodScore,
    this.energy / 10,
    this.productivity / 10,
    (11 - this.stress) / 10,
    this.satisfaction / 10
  ].filter(score => score !== undefined && !isNaN(score));
  
  return scores.length > 0 
    ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) 
    : 0;
});

// Virtual for mood score
lifeEntrySchema.virtual('moodScore').get(function() {
  const moodMap = {
    'terrible': 0.1, 'bad': 0.3, 'okay': 0.5,
    'good': 0.7, 'great': 0.9, 'excellent': 1.0
  };
  return moodMap[this.mood] || 0.5;
});

// Indexes for better query performance
lifeEntrySchema.index({ user: 1, createdAt: -1 });
lifeEntrySchema.index({ prompt: 1, createdAt: -1 });
lifeEntrySchema.index({ mood: 1, createdAt: -1 });
lifeEntrySchema.index({ tags: 1 });
lifeEntrySchema.index({ 'businessMetrics.revenue': -1 });
lifeEntrySchema.index({ 'personalMetrics.exercise': -1 });
lifeEntrySchema.index({ location: '2dsphere' });

// Static method to find entries by user and date range
lifeEntrySchema.statics.findByUserAndDateRange = function(userId, startDate, endDate) {
  return this.find({
    user: userId,
    createdAt: {
      $gte: startDate,
      $lte: endDate
    }
  }).sort({ createdAt: -1 }).populate('prompt');
};

// Static method to find entries by mood
lifeEntrySchema.statics.findByMood = function(userId, mood) {
  return this.find({ user: userId, mood }).sort({ createdAt: -1 });
};

// Static method to get user statistics
lifeEntrySchema.statics.getUserStats = async function(userId) {
  const stats = await this.aggregate([
    { $match: { user: mongoose.Types.ObjectId(userId) } },
    {
      $group: {
        _id: null,
        totalEntries: { $sum: 1 },
        avgMood: { $avg: { $indexOfArray: ['terrible', 'bad', 'okay', 'good', 'great', 'excellent'], '$mood' } },
        avgEnergy: { $avg: '$energy' },
        avgProductivity: { $avg: '$productivity' },
        avgStress: { $avg: '$stress' },
        avgSatisfaction: { $avg: '$satisfaction' },
        totalRevenue: { $sum: '$businessMetrics.revenue' },
        totalExercise: { $sum: '$personalMetrics.exercise' }
      }
    }
  ]);
  
  return stats[0] || {};
};

// Method to add insight
lifeEntrySchema.methods.addInsight = function(insight) {
  if (insight && insight.trim()) {
    this.insights.push(insight.trim());
  }
  return this.save();
};

// Method to add action item
lifeEntrySchema.methods.addAction = function(action) {
  if (action && action.title) {
    this.actions.push(action);
  }
  return this.save();
};

// Method to mark action as completed
lifeEntrySchema.methods.completeAction = function(actionId) {
  const action = this.actions.id(actionId);
  if (action) {
    action.completed = true;
  }
  return this.save();
};

module.exports = mongoose.model('LifeEntry', lifeEntrySchema);
