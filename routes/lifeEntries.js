const express = require('express');
const { body, validationResult, query } = require('express-validator');
const LifeEntry = require('../models/LifeEntry');
const Prompt = require('../models/Prompt');
const User = require('../models/User');
const { protect } = require('../middleware/auth');

const router = express.Router();

// @desc    Get all life entries for a user
// @route   GET /api/life-entries
// @access  Private
router.get('/', protect, [
  query('startDate').optional().isISO8601().withMessage('Start date must be a valid ISO date'),
  query('endDate').optional().isISO8601().withMessage('End date must be a valid ISO date'),
  query('mood').optional().isIn(['terrible', 'bad', 'okay', 'good', 'great', 'excellent']),
  query('category').optional().isString(),
  query('limit').optional().isInt({ min: 1, max: 100 }),
  query('page').optional().isInt({ min: 1 })
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const {
      startDate,
      endDate,
      mood,
      category,
      limit = 20,
      page = 1
    } = req.query;

    // Build filter object
    const filter = { user: req.user._id };
    
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }
    
    if (mood) filter.mood = mood;
    if (category) filter['prompt.category'] = category;

    // Calculate skip value for pagination
    const skip = (page - 1) * limit;

    // Get entries with pagination
    const entries = await LifeEntry.find(filter)
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(skip)
      .populate('prompt', 'title category businessRelevance');

    // Get total count for pagination
    const total = await LifeEntry.countDocuments(filter);

    // Calculate pagination info
    const totalPages = Math.ceil(total / limit);
    const hasNextPage = page < totalPages;
    const hasPrevPage = page > 1;

    res.json({
      success: true,
      data: entries,
      pagination: {
        currentPage: parseInt(page),
        totalPages,
        totalItems: total,
        itemsPerPage: parseInt(limit),
        hasNextPage,
        hasPrevPage
      }
    });
  } catch (error) {
    console.error('Get life entries error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching life entries'
    });
  }
});

// @desc    Get single life entry by ID
// @route   GET /api/life-entries/:id
// @access  Private
router.get('/:id', protect, async (req, res) => {
  try {
    const entry = await LifeEntry.findById(req.params.id)
      .populate('prompt', 'title category content businessRelevance')
      .populate('user', 'firstName lastName');

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry or if it's public
    if (entry.user._id.toString() !== req.user._id.toString() && entry.isPrivate) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to view this entry'
      });
    }

    res.json({
      success: true,
      data: entry
    });
  } catch (error) {
    console.error('Get life entry error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching life entry'
    });
  }
});

// @desc    Create new life entry
// @route   POST /api/life-entries
// @access  Private
router.post('/', protect, [
  body('promptId').isMongoId().withMessage('Valid prompt ID is required'),
  body('title').trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('content').trim().isLength({ min: 10, max: 10000 }).withMessage('Content must be between 10 and 10000 characters'),
  body('mood').isIn(['terrible', 'bad', 'okay', 'good', 'great', 'excellent']).withMessage('Valid mood is required'),
  body('energy').isInt({ min: 1, max: 10 }).withMessage('Energy must be between 1 and 10'),
  body('productivity').optional().isInt({ min: 1, max: 10 }),
  body('stress').optional().isInt({ min: 1, max: 10 }),
  body('satisfaction').optional().isInt({ min: 1, max: 10 })
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const {
      promptId,
      title,
      content,
      mood,
      energy,
      productivity,
      stress,
      satisfaction,
      insights,
      actions,
      gratitude,
      challenges,
      wins,
      businessMetrics,
      personalMetrics,
      tags,
      isPrivate,
      completionTime,
      rating,
      reflection
    } = req.body;

    // Verify prompt exists
    const prompt = await Prompt.findById(promptId);
    if (!prompt || !prompt.isActive) {
      return res.status(400).json({
        success: false,
        message: 'Invalid or inactive prompt'
      });
    }

    // Create life entry
    const entryData = {
      user: req.user._id,
      prompt: promptId,
      title,
      content,
      mood,
      energy,
      productivity,
      stress,
      satisfaction,
      insights: insights || [],
      actions: actions || [],
      gratitude: gratitude || [],
      challenges: challenges || [],
      wins: wins || [],
      businessMetrics: businessMetrics || {},
      personalMetrics: personalMetrics || {},
      tags: tags || [],
      isPrivate: isPrivate || false,
      completionTime,
      rating,
      reflection: reflection || {}
    };

    const entry = await LifeEntry.create(entryData);

    // Increment prompt usage count
    await prompt.incrementUsage();

    // Update user stats
    await User.findByIdAndUpdate(req.user._id, {
      $inc: { 'stats.totalEntries': 1 },
      $set: { 'stats.lastEntryDate': new Date() }
    });

    // Populate prompt details for response
    await entry.populate('prompt', 'title category businessRelevance');

    res.status(201).json({
      success: true,
      data: entry
    });
  } catch (error) {
    console.error('Create life entry error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while creating life entry'
    });
  }
});

// @desc    Update life entry
// @route   PUT /api/life-entries/:id
// @access  Private
router.put('/:id', protect, [
  body('title').optional().trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('content').optional().trim().isLength({ min: 10, max: 10000 }).withMessage('Content must be between 10 and 10000 characters'),
  body('mood').optional().isIn(['terrible', 'bad', 'okay', 'good', 'great', 'excellent']).withMessage('Valid mood is required'),
  body('energy').optional().isInt({ min: 1, max: 10 }).withMessage('Energy must be between 1 and 10')
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const entry = await LifeEntry.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry
    if (entry.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to update this entry'
      });
    }

    // Update entry
    Object.keys(req.body).forEach(key => {
      if (key !== 'user' && key !== 'prompt' && key !== '_id') {
        entry[key] = req.body[key];
      }
    });

    const updatedEntry = await entry.save();

    // Populate prompt details for response
    await updatedEntry.populate('prompt', 'title category businessRelevance');

    res.json({
      success: true,
      data: updatedEntry
    });
  } catch (error) {
    console.error('Update life entry error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating life entry'
    });
  }
});

// @desc    Delete life entry
// @route   DELETE /api/life-entries/:id
// @access  Private
router.delete('/:id', protect, async (req, res) => {
  try {
    const entry = await LifeEntry.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry
    if (entry.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this entry'
      });
    }

    await LifeEntry.findByIdAndDelete(req.params.id);

    // Update user stats
    await User.findByIdAndUpdate(req.user._id, {
      $inc: { 'stats.totalEntries': -1 }
    });

    res.json({
      success: true,
      message: 'Life entry deleted successfully'
    });
  } catch (error) {
    console.error('Delete life entry error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while deleting life entry'
    });
  }
});

// @desc    Add insight to life entry
// @route   POST /api/life-entries/:id/insights
// @access  Private
router.post('/:id/insights', protect, [
  body('insight').trim().isLength({ min: 1, max: 500 }).withMessage('Insight must be between 1 and 500 characters')
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { insight } = req.body;
    const entry = await LifeEntry.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry
    if (entry.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to modify this entry'
      });
    }

    await entry.addInsight(insight);

    res.json({
      success: true,
      data: entry,
      message: 'Insight added successfully'
    });
  } catch (error) {
    console.error('Add insight error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while adding insight'
    });
  }
});

// @desc    Add action to life entry
// @route   POST /api/life-entries/:id/actions
// @access  Private
router.post('/:id/actions', protect, [
  body('title').trim().isLength({ min: 1, max: 200 }).withMessage('Action title must be between 1 and 200 characters'),
  body('description').optional().trim().isLength({ max: 1000 }).withMessage('Description cannot exceed 1000 characters'),
  body('dueDate').optional().isISO8601().withMessage('Due date must be a valid ISO date'),
  body('priority').optional().isIn(['low', 'medium', 'high', 'urgent'])
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const { title, description, dueDate, priority } = req.body;
    const entry = await LifeEntry.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry
    if (entry.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to modify this entry'
      });
    }

    const action = { title, description, dueDate, priority };
    await entry.addAction(action);

    res.json({
      success: true,
      data: entry,
      message: 'Action added successfully'
    });
  } catch (error) {
    console.error('Add action error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while adding action'
    });
  }
});

// @desc    Complete action in life entry
// @route   PUT /api/life-entries/:id/actions/:actionId/complete
// @access  Private
router.put('/:id/actions/:actionId/complete', protect, async (req, res) => {
  try {
    const entry = await LifeEntry.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({
        success: false,
        message: 'Life entry not found'
      });
    }

    // Check if user owns this entry
    if (entry.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to modify this entry'
      });
    }

    await entry.completeAction(req.params.actionId);

    res.json({
      success: true,
      data: entry,
      message: 'Action completed successfully'
    });
  } catch (error) {
    console.error('Complete action error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while completing action'
    });
  }
});

// @desc    Get user statistics
// @route   GET /api/life-entries/stats/overview
// @access  Private
router.get('/stats/overview', protect, async (req, res) => {
  try {
    const stats = await LifeEntry.getUserStats(req.user._id);

    res.json({
      success: true,
      data: stats
    });
  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching statistics'
    });
  }
});

// @desc    Get entries by mood
// @route   GET /api/life-entries/mood/:mood
// @access  Private
router.get('/mood/:mood', protect, [
  query('limit').optional().isInt({ min: 1, max: 100 }),
  query('page').optional().isInt({ min: 1 })
], async (req, res) => {
  try {
    const { mood } = req.params;
    const { limit = 20, page = 1 } = req.query;

    if (!['terrible', 'bad', 'okay', 'good', 'great', 'excellent'].includes(mood)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid mood value'
      });
    }

    const entries = await LifeEntry.findByMood(req.user._id, mood)
      .limit(parseInt(limit))
      .skip((page - 1) * limit)
      .populate('prompt', 'title category');

    const total = await LifeEntry.countDocuments({ user: req.user._id, mood });

    res.json({
      success: true,
      data: entries,
      pagination: {
        currentPage: parseInt(page),
        totalPages: Math.ceil(total / limit),
        totalItems: total,
        itemsPerPage: parseInt(limit)
      }
    });
  } catch (error) {
    console.error('Get entries by mood error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching entries by mood'
    });
  }
});

module.exports = router;
