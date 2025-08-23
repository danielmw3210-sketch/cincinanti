const express = require('express');
const { body, validationResult, query } = require('express-validator');
const Goal = require('../models/Goal');
const { protect } = require('../middleware/auth');

const router = express.Router();

// @desc    Get all goals for a user
// @route   GET /api/goals
// @access  Private
router.get('/', protect, [
  query('status').optional().isIn(['not-started', 'in-progress', 'on-hold', 'completed', 'cancelled']),
  query('category').optional().isString(),
  query('priority').optional().isIn(['low', 'medium', 'high', 'urgent']),
  query('type').optional().isIn(['short-term', 'medium-term', 'long-term']),
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
      status,
      category,
      priority,
      type,
      limit = 20,
      page = 1
    } = req.query;

    // Build filter object
    const filter = { user: req.user._id };
    if (status) filter.status = status;
    if (category) filter.category = category;
    if (priority) filter.priority = priority;
    if (type) filter.type = type;

    // Calculate skip value for pagination
    const skip = (page - 1) * limit;

    // Get goals with pagination
    const goals = await Goal.find(filter)
      .sort({ targetDate: 1, priority: -1 })
      .limit(parseInt(limit))
      .skip(skip);

    // Get total count for pagination
    const total = await Goal.countDocuments(filter);

    // Calculate pagination info
    const totalPages = Math.ceil(total / limit);
    const hasNextPage = page < totalPages;
    const hasPrevPage = page > 1;

    res.json({
      success: true,
      data: goals,
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
    console.error('Get goals error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching goals'
    });
  }
});

// @desc    Get single goal by ID
// @route   GET /api/goals/:id
// @access  Private
router.get('/:id', protect, async (req, res) => {
  try {
    const goal = await Goal.findById(req.params.id);

    if (!goal) {
      return res.status(404).json({
        success: false,
        message: 'Goal not found'
      });
    }

    // Check if user owns this goal
    if (goal.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to view this goal'
      });
    }

    res.json({
      success: true,
      data: goal
    });
  } catch (error) {
    console.error('Get goal error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching goal'
    });
  }
});

// @desc    Create new goal
// @route   POST /api/goals
// @access  Private
router.post('/', protect, [
  body('title').trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('description').trim().isLength({ min: 10, max: 1000 }).withMessage('Description must be between 10 and 1000 characters'),
  body('category').isIn([
    'business', 'personal', 'health', 'finance', 'relationships',
    'career', 'education', 'spirituality', 'creativity', 'travel', 'home', 'community'
  ]).withMessage('Invalid category'),
  body('type').isIn(['short-term', 'medium-term', 'long-term']).withMessage('Invalid goal type'),
  body('targetDate').isISO8601().withMessage('Target date must be a valid ISO date'),
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

    const goalData = {
      ...req.body,
      user: req.user._id
    };

    const goal = await Goal.create(goalData);

    res.status(201).json({
      success: true,
      data: goal
    });
  } catch (error) {
    console.error('Create goal error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while creating goal'
    });
  }
});

// @desc    Update goal
// @route   PUT /api/goals/:id
// @access  Private
router.put('/:id', protect, [
  body('title').optional().trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('description').optional().trim().isLength({ min: 10, max: 1000 }).withMessage('Description must be between 10 and 1000 characters')
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

    const goal = await Goal.findById(req.params.id);

    if (!goal) {
      return res.status(404).json({
        success: false,
        message: 'Goal not found'
      });
    }

    // Check if user owns this goal
    if (goal.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to update this goal'
      });
    }

    // Update goal
    Object.keys(req.body).forEach(key => {
      if (key !== 'user' && key !== '_id') {
        goal[key] = req.body[key];
      }
    });

    const updatedGoal = await goal.save();

    res.json({
      success: true,
      data: updatedGoal
    });
  } catch (error) {
    console.error('Update goal error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating goal'
    });
  }
});

// @desc    Delete goal
// @route   DELETE /api/goals/:id
// @access  Private
router.delete('/:id', protect, async (req, res) => {
  try {
    const goal = await Goal.findById(req.params.id);

    if (!goal) {
      return res.status(404).json({
        success: false,
        message: 'Goal not found'
      });
    }

    // Check if user owns this goal
    if (goal.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this goal'
      });
    }

    await Goal.findByIdAndDelete(req.params.id);

    res.json({
      success: true,
      message: 'Goal deleted successfully'
    });
  } catch (error) {
    console.error('Delete goal error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while deleting goal'
    });
  }
});

// @desc    Update goal progress
// @route   PUT /api/goals/:id/progress
// @access  Private
router.put('/:id/progress', protect, [
  body('progress').isInt({ min: 0, max: 100 }).withMessage('Progress must be between 0 and 100')
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

    const { progress } = req.body;
    const goal = await Goal.findById(req.params.id);

    if (!goal) {
      return res.status(404).json({
        success: false,
        message: 'Goal not found'
      });
    }

    // Check if user owns this goal
    if (goal.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to update this goal'
      });
    }

    await goal.updateProgress(progress);

    res.json({
      success: true,
      data: goal,
      message: 'Progress updated successfully'
    });
  } catch (error) {
    console.error('Update progress error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating progress'
    });
  }
});

// @desc    Get overdue goals
// @route   GET /api/goals/overdue
// @access  Private
router.get('/overdue', protect, async (req, res) => {
  try {
    const goals = await Goal.findOverdue(req.user._id);

    res.json({
      success: true,
      data: goals
    });
  } catch (error) {
    console.error('Get overdue goals error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching overdue goals'
    });
  }
});

// @desc    Get goals by category
// @route   GET /api/goals/category/:category
// @access  Private
router.get('/category/:category', protect, async (req, res) => {
  try {
    const { category } = req.params;
    const goals = await Goal.findByCategory(req.user._id, category);

    res.json({
      success: true,
      data: goals
    });
  } catch (error) {
    console.error('Get goals by category error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching goals by category'
    });
  }
});

module.exports = router;
