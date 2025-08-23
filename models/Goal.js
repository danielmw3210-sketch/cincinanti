const mongoose = require('mongoose');

const goalSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  title: {
    type: String,
    required: [true, 'Goal title is required'],
    trim: true,
    maxlength: [200, 'Title cannot be more than 200 characters']
  },
  description: {
    type: String,
    required: [true, 'Goal description is required'],
    trim: true,
    maxlength: [1000, 'Description cannot be more than 1000 characters']
  },
  category: {
    type: String,
    required: [true, 'Goal category is required'],
    enum: [
      'business',
      'personal',
      'health',
      'finance',
      'relationships',
      'career',
      'education',
      'spirituality',
      'creativity',
      'travel',
      'home',
      'community'
    ]
  },
  subcategory: String,
  type: {
    type: String,
    enum: ['short-term', 'medium-term', 'long-term'],
    required: true
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  status: {
    type: String,
    enum: ['not-started', 'in-progress', 'on-hold', 'completed', 'cancelled'],
    default: 'not-started'
  },
  progress: {
    type: Number,
    min: 0,
    max: 100,
    default: 0
  },
  targetDate: {
    type: Date,
    required: true
  },
  startDate: {
    type: Date,
    default: Date.now
  },
  completedDate: Date,
  milestones: [{
    title: String,
    description: String,
    targetDate: Date,
    completed: { type: Boolean, default: false },
    completedDate: Date,
    progress: { type: Number, min: 0, max: 100, default: 0 }
  }],
  metrics: {
    target: Number,
    current: Number,
    unit: String
  },
  actions: [{
    title: String,
    description: String,
    dueDate: Date,
    completed: { type: Boolean, default: false },
    completedDate: Date,
    priority: {
      type: String,
      enum: ['low', 'medium', 'high', 'urgent'],
      default: 'medium'
    }
  }],
  resources: [{
    title: String,
    url: String,
    type: {
      type: String,
      enum: ['article', 'video', 'book', 'podcast', 'tool', 'person', 'other']
    },
    description: String,
    cost: Number
  }],
  obstacles: [{
    description: String,
    impact: {
      type: String,
      enum: ['low', 'medium', 'high', 'critical']
    },
    mitigation: String,
    status: {
      type: String,
      enum: ['active', 'resolved', 'mitigated'],
      default: 'active'
    }
  }],
  support: [{
    name: String,
    role: String,
    contact: String,
    commitment: String
  }],
  motivation: {
    why: String,
    benefits: [String],
    consequences: [String]
  },
  reflection: {
    whatILearned: String,
    whatWentWell: String,
    whatCouldBeBetter: String,
    nextSteps: String
  },
  tags: [String],
  isPublic: {
    type: Boolean,
    default: false
  },
  estimatedEffort: {
    type: String,
    enum: ['low', 'medium', 'high', 'extreme']
  },
  estimatedCost: {
    amount: Number,
    currency: { type: String, default: 'USD' }
  },
  businessImpact: {
    revenue: { type: Number, default: 0 },
    customers: { type: Number, default: 0 },
    efficiency: { type: Number, min: 0, max: 100, default: 0 },
    reputation: { type: Number, min: 0, max: 100, default: 0 }
  },
  personalImpact: {
    happiness: { type: Number, min: 0, max: 100, default: 0 },
    health: { type: Number, min: 0, max: 100, default: 0 },
    relationships: { type: Number, min: 0, max: 100, default: 0 },
    growth: { type: Number, min: 0, max: 100, default: 0 }
  },
  attachments: [{
    filename: String,
    originalName: String,
    mimeType: String,
    size: Number,
    url: String
  }],
  notes: [{
    content: String,
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now }
  }]
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for goal urgency
goalSchema.virtual('urgency').get(function() {
  const daysUntilTarget = Math.ceil((this.targetDate - new Date()) / (1000 * 60 * 60 * 24));
  if (daysUntilTarget < 0) return 'overdue';
  if (daysUntilTarget <= 7) return 'urgent';
  if (daysUntilTarget <= 30) return 'soon';
  if (daysUntilTarget <= 90) return 'upcoming';
  return 'distant';
});

// Virtual for goal health
goalSchema.virtual('health').get(function() {
  const daysUntilTarget = Math.ceil((this.targetDate - new Date()) / (1000 * 60 * 60 * 24));
  const daysElapsed = Math.ceil((new Date() - this.startDate) / (1000 * 60 * 60 * 24));
  const totalDays = Math.ceil((this.targetDate - this.startDate) / (1000 * 60 * 60 * 24));
  
  if (daysUntilTarget < 0) return 'overdue';
  if (totalDays <= 0) return 'invalid';
  
  const expectedProgress = (daysElapsed / totalDays) * 100;
  if (this.progress >= expectedProgress) return 'on-track';
  if (this.progress >= expectedProgress * 0.8) return 'slightly-behind';
  return 'behind';
});

// Virtual for completion status
goalSchema.virtual('isOverdue').get(function() {
  return this.targetDate < new Date() && this.status !== 'completed';
});

// Indexes for better query performance
goalSchema.index({ user: 1, status: 1 });
goalSchema.index({ user: 1, category: 1 });
goalSchema.index({ user: 1, targetDate: 1 });
goalSchema.index({ user: 1, priority: 1 });
goalSchema.index({ status: 1, targetDate: 1 });
goalSchema.index({ tags: 1 });

// Static method to find goals by user and status
goalSchema.statics.findByUserAndStatus = function(userId, status) {
  return this.find({ user: userId, status }).sort({ targetDate: 1 });
};

// Static method to find overdue goals
goalSchema.statics.findOverdue = function(userId) {
  return this.find({
    user: userId,
    targetDate: { $lt: new Date() },
    status: { $ne: 'completed' }
  }).sort({ targetDate: 1 });
};

// Static method to find goals by category
goalSchema.statics.findByCategory = function(userId, category) {
  return this.find({ user: userId, category }).sort({ targetDate: 1 });
};

// Method to update progress
goalSchema.methods.updateProgress = function(newProgress) {
  if (newProgress >= 0 && newProgress <= 100) {
    this.progress = newProgress;
    if (newProgress === 100 && this.status !== 'completed') {
      this.status = 'completed';
      this.completedDate = new Date();
    }
  }
  return this.save();
};

// Method to add milestone
goalSchema.methods.addMilestone = function(milestone) {
  if (milestone && milestone.title) {
    this.milestones.push(milestone);
  }
  return this.save();
};

// Method to complete milestone
goalSchema.methods.completeMilestone = function(milestoneId) {
  const milestone = this.milestones.id(milestoneId);
  if (milestone) {
    milestone.completed = true;
    milestone.completedDate = new Date();
    milestone.progress = 100;
  }
  return this.save();
};

// Method to add action
goalSchema.methods.addAction = function(action) {
  if (action && action.title) {
    this.actions.push(action);
  }
  return this.save();
};

// Method to complete action
goalSchema.methods.completeAction = function(actionId) {
  const action = this.actions.id(actionId);
  if (action) {
    action.completed = true;
    action.completedDate = new Date();
  }
  return this.save();
};

// Method to add note
goalSchema.methods.addNote = function(content) {
  if (content && content.trim()) {
    this.notes.push({ content: content.trim() });
  }
  return this.save();
};

module.exports = mongoose.model('Goal', goalSchema);
