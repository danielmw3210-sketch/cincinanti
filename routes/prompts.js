const express = require('express');
const { body, validationResult, query } = require('express-validator');
const Prompt = require('../models/Prompt');
const { protect, optionalAuth } = require('../middleware/auth');

const router = express.Router();

// @desc    Get all prompts with filtering
// @route   GET /api/prompts
// @access  Public (with optional auth)
router.get('/', optionalAuth, [
  query('category').optional().isIn([
    'business', 'personal', 'health', 'relationships', 'finance',
    'spirituality', 'creativity', 'learning', 'productivity',
    'mindfulness', 'gratitude', 'reflection', 'planning',
    'challenge', 'celebration'
  ]),
  query('difficulty').optional().isIn(['easy', 'medium', 'hard', 'expert']),
  query('businessRelevance').optional().isIn(['low', 'medium', 'high', 'critical']),
  query('promptType').optional().isIn(['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'one-time', 'recurring']),
  query('featured').optional().isBoolean(),
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
      category,
      difficulty,
      businessRelevance,
      promptType,
      featured,
      tags,
      limit = 20,
      page = 1,
      sort = 'averageRating'
    } = req.query;

    // Build filter object
    const filter = { isActive: true };
    if (category) filter.category = category;
    if (difficulty) filter.difficulty = difficulty;
    if (businessRelevance) filter.businessRelevance = businessRelevance;
    if (promptType) filter.promptType = promptType;
    if (featured === 'true') filter.isFeatured = true;
    if (tags) {
      const tagArray = tags.split(',').map(tag => tag.trim());
      filter.tags = { $in: tagArray };
    }

    // Build sort object
    let sortObj = {};
    if (sort === 'popularity') {
      sortObj = { usageCount: -1, averageRating: -1 };
    } else if (sort === 'newest') {
      sortObj = { createdAt: -1 };
    } else if (sort === 'businessPriority') {
      sortObj = { businessRelevance: -1, averageRating: -1 };
    } else {
      sortObj = { averageRating: -1, usageCount: -1 };
    }

    // Calculate skip value for pagination
    const skip = (page - 1) * limit;

    // Get prompts with pagination
    const prompts = await Prompt.find(filter)
      .sort(sortObj)
      .limit(parseInt(limit))
      .skip(skip)
      .populate('createdBy', 'firstName lastName');

    // Get total count for pagination
    const total = await Prompt.countDocuments(filter);

    // Calculate pagination info
    const totalPages = Math.ceil(total / limit);
    const hasNextPage = page < totalPages;
    const hasPrevPage = page > 1;

    res.json({
      success: true,
      data: prompts,
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
    console.error('Get prompts error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching prompts'
    });
  }
});

// @desc    Get single prompt by ID
// @route   GET /api/prompts/:id
// @access  Public
router.get('/:id', async (req, res) => {
  try {
    const prompt = await Prompt.findById(req.params.id)
      .populate('createdBy', 'firstName lastName');

    if (!prompt) {
      return res.status(404).json({
        success: false,
        message: 'Prompt not found'
      });
    }

    if (!prompt.isActive) {
      return res.status(404).json({
        success: false,
        message: 'Prompt is not available'
      });
    }

    res.json({
      success: true,
      data: prompt
    });
  } catch (error) {
    console.error('Get prompt error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching prompt'
    });
  }
});

// @desc    Create new prompt
// @route   POST /api/prompts
// @access  Private
router.post('/', protect, [
  body('title').trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('content').trim().isLength({ min: 10, max: 2000 }).withMessage('Content must be between 10 and 2000 characters'),
  body('category').isIn([
    'business', 'personal', 'health', 'relationships', 'finance',
    'spirituality', 'creativity', 'learning', 'productivity',
    'mindfulness', 'gratitude', 'reflection', 'planning',
    'challenge', 'celebration'
  ]).withMessage('Invalid category'),
  body('difficulty').optional().isIn(['easy', 'medium', 'hard', 'expert']),
  body('estimatedTime').optional().isInt({ min: 1, max: 480 }),
  body('businessRelevance').optional().isIn(['low', 'medium', 'high', 'critical']),
  body('promptType').optional().isIn(['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'one-time', 'recurring'])
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

    const promptData = {
      ...req.body,
      createdBy: req.user._id
    };

    const prompt = await Prompt.create(promptData);

    res.status(201).json({
      success: true,
      data: prompt
    });
  } catch (error) {
    console.error('Create prompt error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while creating prompt'
    });
  }
});

// @desc    Update prompt
// @route   PUT /api/prompts/:id
// @access  Private (creator only)
router.put('/:id', protect, [
  body('title').optional().trim().isLength({ min: 5, max: 200 }).withMessage('Title must be between 5 and 200 characters'),
  body('content').optional().trim().isLength({ min: 10, max: 2000 }).withMessage('Content must be between 10 and 2000 characters'),
  body('category').optional().isIn([
    'business', 'personal', 'health', 'relationships', 'finance',
    'spirituality', 'creativity', 'learning', 'productivity',
    'mindfulness', 'gratitude', 'reflection', 'planning',
    'challenge', 'celebration'
  ]).withMessage('Invalid category')
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

    const prompt = await Prompt.findById(req.params.id);

    if (!prompt) {
      return res.status(404).json({
        success: false,
        message: 'Prompt not found'
      });
    }

    // Check if user is the creator
    if (prompt.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to update this prompt'
      });
    }

    // Update prompt
    Object.keys(req.body).forEach(key => {
      if (key !== 'createdBy' && key !== '_id') {
        prompt[key] = req.body[key];
      }
    });

    const updatedPrompt = await prompt.save();

    res.json({
      success: true,
      data: updatedPrompt
    });
  } catch (error) {
    console.error('Update prompt error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating prompt'
    });
  }
});

// @desc    Delete prompt
// @route   DELETE /api/prompts/:id
// @access  Private (creator only)
router.delete('/:id', protect, async (req, res) => {
  try {
    const prompt = await Prompt.findById(req.params.id);

    if (!prompt) {
      return res.status(404).json({
        success: false,
        message: 'Prompt not found'
      });
    }

    // Check if user is the creator
    if (prompt.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this prompt'
      });
    }

    // Soft delete by setting isActive to false
    prompt.isActive = false;
    await prompt.save();

    res.json({
      success: true,
      message: 'Prompt deleted successfully'
    });
  } catch (error) {
    console.error('Delete prompt error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while deleting prompt'
    });
  }
});

// @desc    Rate a prompt
// @route   POST /api/prompts/:id/rate
// @access  Private
router.post('/:id/rate', protect, [
  body('rating').isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1 and 5')
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

    const { rating } = req.body;
    const prompt = await Prompt.findById(req.params.id);

    if (!prompt) {
      return res.status(404).json({
        success: false,
        message: 'Prompt not found'
      });
    }

    if (!prompt.isActive) {
      return res.status(404).json({
        success: false,
        message: 'Prompt is not available'
      });
    }

    // Add rating
    await prompt.addRating(rating);

    res.json({
      success: true,
      message: 'Rating added successfully',
      data: {
        averageRating: prompt.averageRating,
        totalRatings: prompt.totalRatings
      }
    });
  } catch (error) {
    console.error('Rate prompt error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while rating prompt'
    });
  }
});

// @desc    Get featured prompts
// @route   GET /api/prompts/featured
// @access  Public
router.get('/featured', async (req, res) => {
  try {
    const prompts = await Prompt.findFeatured().limit(10);

    res.json({
      success: true,
      data: prompts
    });
  } catch (error) {
    console.error('Get featured prompts error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching featured prompts'
    });
  }
});

// @desc    Get prompts by business relevance
// @route   GET /api/prompts/business/:relevance
// @access  Public
router.get('/business/:relevance', async (req, res) => {
  try {
    const { relevance } = req.params;
    
    if (!['low', 'medium', 'high', 'critical'].includes(relevance)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid business relevance level'
      });
    }

    const prompts = await Prompt.findByBusinessRelevance(relevance).limit(20);

    res.json({
      success: true,
      data: prompts
    });
  } catch (error) {
    console.error('Get business prompts error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching business prompts'
    });
  }
});

module.exports = router;
