const express = require('express');
const { query } = require('express-validator');
const LifeEntry = require('../models/LifeEntry');
const Goal = require('../models/Goal');
const { protect } = require('../middleware/auth');

const router = express.Router();

// @desc    Get user analytics overview
// @route   GET /api/analytics/overview
// @access  Private
router.get('/overview', protect, [
  query('startDate').optional().isISO8601().withMessage('Start date must be a valid ISO date'),
  query('endDate').optional().isISO8601().withMessage('End date must be a valid ISO date')
], async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    
    // Build date filter
    const dateFilter = {};
    if (startDate || endDate) {
      if (startDate) dateFilter.$gte = new Date(startDate);
      if (endDate) dateFilter.$lte = new Date(endDate);
    }

    // Get life entries in date range
    const entries = await LifeEntry.find({
      user: req.user._id,
      ...(Object.keys(dateFilter).length > 0 && { createdAt: dateFilter })
    });

    // Get goals
    const goals = await Goal.find({ user: req.user._id });

    // Calculate analytics
    const analytics = {
      totalEntries: entries.length,
      totalGoals: goals.length,
      completedGoals: goals.filter(g => g.status === 'completed').length,
      activeGoals: goals.filter(g => g.status === 'in-progress').length,
      overdueGoals: goals.filter(g => g.isOverdue).length,
      averageMood: calculateAverageMood(entries),
      moodTrend: calculateMoodTrend(entries),
      energyTrend: calculateEnergyTrend(entries),
      productivityTrend: calculateProductivityTrend(entries),
      businessMetrics: calculateBusinessMetrics(entries),
      personalMetrics: calculatePersonalMetrics(entries),
      topInsights: getTopInsights(entries),
      topActions: getTopActions(entries),
      categoryBreakdown: getCategoryBreakdown(entries),
      streakInfo: calculateStreakInfo(entries)
    };

    res.json({
      success: true,
      data: analytics
    });
  } catch (error) {
    console.error('Get analytics error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching analytics'
    });
  }
});

// @desc    Get mood analytics
// @route   GET /api/analytics/mood
// @access  Private
router.get('/mood', protect, [
  query('startDate').optional().isISO8601().withMessage('Start date must be a valid ISO date'),
  query('endDate').optional().isISO8601().withMessage('End date must be a valid ISO date')
], async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    
    const dateFilter = {};
    if (startDate || endDate) {
      if (startDate) dateFilter.$gte = new Date(startDate);
      if (endDate) dateFilter.$lte = new Date(endDate);
    }

    const entries = await LifeEntry.find({
      user: req.user._id,
      ...(Object.keys(dateFilter).length > 0 && { createdAt: dateFilter })
    });

    const moodData = {
      distribution: getMoodDistribution(entries),
      trend: getMoodTrend(entries),
      factors: analyzeMoodFactors(entries),
      weeklyPattern: getWeeklyMoodPattern(entries),
      monthlyPattern: getMonthlyMoodPattern(entries)
    };

    res.json({
      success: true,
      data: moodData
    });
  } catch (error) {
    console.error('Get mood analytics error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching mood analytics'
    });
  }
});

// @desc    Get business analytics
// @route   GET /api/analytics/business
// @access  Private
router.get('/business', protect, [
  query('startDate').optional().isISO8601().withMessage('Start date must be a valid ISO date'),
  query('endDate').optional().isISO8601().withMessage('End date must be a valid ISO date')
], async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    
    const dateFilter = {};
    if (startDate || endDate) {
      if (startDate) dateFilter.$gte = new Date(startDate);
      if (endDate) dateFilter.$lte = new Date(endDate);
    }

    const entries = await LifeEntry.find({
      user: req.user._id,
      ...(Object.keys(dateFilter).length > 0 && { createdAt: dateFilter })
    });

    const businessData = {
      revenue: calculateRevenueMetrics(entries),
      productivity: calculateProductivityMetrics(entries),
      customerMetrics: calculateCustomerMetrics(entries),
      projectMetrics: calculateProjectMetrics(entries),
      meetingMetrics: calculateMeetingMetrics(entries),
      decisionMetrics: calculateDecisionMetrics(entries),
      innovationMetrics: calculateInnovationMetrics(entries),
      trends: calculateBusinessTrends(entries)
    };

    res.json({
      success: true,
      data: businessData
    });
  } catch (error) {
    console.error('Get business analytics error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching business analytics'
    });
  }
});

// Helper functions for analytics calculations
function calculateAverageMood(entries) {
  if (entries.length === 0) return 0;
  
  const moodScores = entries.map(entry => {
    const moodMap = {
      'terrible': 1, 'bad': 2, 'okay': 3,
      'good': 4, 'great': 5, 'excellent': 6
    };
    return moodMap[entry.mood] || 3;
  });
  
  return Math.round(moodScores.reduce((a, b) => a + b, 0) / moodScores.length);
}

function calculateMoodTrend(entries) {
  if (entries.length < 2) return 'stable';
  
  const sortedEntries = entries.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
  const recentEntries = sortedEntries.slice(-7); // Last 7 entries
  
  if (recentEntries.length < 2) return 'stable';
  
  const recentAvg = calculateAverageMood(recentEntries);
  const overallAvg = calculateAverageMood(entries);
  
  if (recentAvg > overallAvg + 0.5) return 'improving';
  if (recentAvg < overallAvg - 0.5) return 'declining';
  return 'stable';
}

function calculateEnergyTrend(entries) {
  if (entries.length < 2) return 'stable';
  
  const sortedEntries = entries.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
  const recentEntries = sortedEntries.slice(-7);
  
  if (recentEntries.length < 2) return 'stable';
  
  const recentAvg = recentEntries.reduce((sum, entry) => sum + (entry.energy || 5), 0) / recentEntries.length;
  const overallAvg = entries.reduce((sum, entry) => sum + (entry.energy || 5), 0) / entries.length;
  
  if (recentAvg > overallAvg + 0.5) return 'improving';
  if (recentAvg < overallAvg - 0.5) return 'declining';
  return 'stable';
}

function calculateProductivityTrend(entries) {
  if (entries.length < 2) return 'stable';
  
  const sortedEntries = entries.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
  const recentEntries = sortedEntries.slice(-7);
  
  if (recentEntries.length < 2) return 'stable';
  
  const recentAvg = recentEntries.reduce((sum, entry) => sum + (entry.productivity || 5), 0) / recentEntries.length;
  const overallAvg = entries.reduce((sum, entry) => sum + (entry.productivity || 5), 0) / entries.length;
  
  if (recentAvg > overallAvg + 0.5) return 'improving';
  if (recentAvg < overallAvg - 0.5) return 'declining';
  return 'stable';
}

function calculateBusinessMetrics(entries) {
  const businessEntries = entries.filter(entry => entry.businessMetrics);
  
  return {
    totalRevenue: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.revenue || 0), 0),
    totalCustomers: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.customers || 0), 0),
    totalProjects: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.projects || 0), 0),
    totalMeetings: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.meetings || 0), 0),
    totalDecisions: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.decisions || 0), 0),
    totalInnovations: businessEntries.reduce((sum, entry) => sum + (entry.businessMetrics.innovations || 0), 0)
  };
}

function calculatePersonalMetrics(entries) {
  const personalEntries = entries.filter(entry => entry.personalMetrics);
  
  return {
    totalExercise: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.exercise || 0), 0),
    totalSleep: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.sleep || 0), 0),
    totalReading: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.reading || 0), 0),
    totalMeditation: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.meditation || 0), 0),
    totalSocial: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.social || 0), 0),
    totalLearning: personalEntries.reduce((sum, entry) => sum + (entry.personalMetrics.learning || 0), 0)
  };
}

function getTopInsights(entries) {
  const allInsights = entries.flatMap(entry => entry.insights || []);
  const insightCounts = {};
  
  allInsights.forEach(insight => {
    insightCounts[insight] = (insightCounts[insight] || 0) + 1;
  });
  
  return Object.entries(insightCounts)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 5)
    .map(([insight, count]) => ({ insight, count }));
}

function getTopActions(entries) {
  const allActions = entries.flatMap(entry => entry.actions || []);
  const actionCounts = {};
  
  allActions.forEach(action => {
    const key = action.title;
    actionCounts[key] = (actionCounts[key] || 0) + 1;
  });
  
  return Object.entries(actionCounts)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 5)
    .map(([action, count]) => ({ action, count }));
}

function getCategoryBreakdown(entries) {
  const categoryCounts = {};
  
  entries.forEach(entry => {
    if (entry.prompt && entry.prompt.category) {
      categoryCounts[entry.prompt.category] = (categoryCounts[entry.prompt.category] || 0) + 1;
    }
  });
  
  return Object.entries(categoryCounts)
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => b.count - a.count);
}

function calculateStreakInfo(entries) {
  if (entries.length === 0) return { currentStreak: 0, longestStreak: 0, lastEntryDate: null };
  
  const sortedEntries = entries.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  const lastEntryDate = new Date(sortedEntries[0].createdAt);
  
  let currentStreak = 0;
  let longestStreak = 0;
  let tempStreak = 0;
  
  for (let i = 0; i < sortedEntries.length; i++) {
    const entryDate = new Date(sortedEntries[i].createdAt);
    const prevEntryDate = i > 0 ? new Date(sortedEntries[i-1].createdAt) : null;
    
    if (prevEntryDate) {
      const dayDiff = Math.floor((prevEntryDate - entryDate) / (1000 * 60 * 60 * 24));
      if (dayDiff === 1) {
        tempStreak++;
      } else {
        longestStreak = Math.max(longestStreak, tempStreak);
        tempStreak = 0;
      }
    } else {
      tempStreak = 1;
    }
  }
  
  longestStreak = Math.max(longestStreak, tempStreak);
  
  // Calculate current streak
  const today = new Date();
  const daysSinceLastEntry = Math.floor((today - lastEntryDate) / (1000 * 60 * 60 * 24));
  
  if (daysSinceLastEntry === 0) {
    currentStreak = tempStreak;
  } else if (daysSinceLastEntry === 1) {
    currentStreak = tempStreak;
  } else {
    currentStreak = 0;
  }
  
  return {
    currentStreak,
    longestStreak,
    lastEntryDate
  };
}

// Additional helper functions for specific analytics
function getMoodDistribution(entries) {
  const moodCounts = {};
  entries.forEach(entry => {
    moodCounts[entry.mood] = (moodCounts[entry.mood] || 0) + 1;
  });
  return moodCounts;
}

function getMoodTrend(entries) {
  // Implementation for mood trend over time
  return 'stable'; // Placeholder
}

function analyzeMoodFactors(entries) {
  // Implementation for analyzing what factors affect mood
  return []; // Placeholder
}

function getWeeklyMoodPattern(entries) {
  // Implementation for weekly mood patterns
  return {}; // Placeholder
}

function getMonthlyMoodPattern(entries) {
  // Implementation for monthly mood patterns
  return {}; // Placeholder
}

// Business analytics helper functions
function calculateRevenueMetrics(entries) {
  return { total: 0, average: 0, trend: 'stable' }; // Placeholder
}

function calculateProductivityMetrics(entries) {
  return { average: 0, trend: 'stable' }; // Placeholder
}

function calculateCustomerMetrics(entries) {
  return { total: 0, new: 0, retained: 0 }; // Placeholder
}

function calculateProjectMetrics(entries) {
  return { total: 0, completed: 0, inProgress: 0 }; // Placeholder
}

function calculateMeetingMetrics(entries) {
  return { total: 0, averageDuration: 0, effectiveness: 0 }; // Placeholder
}

function calculateDecisionMetrics(entries) {
  return { total: 0, good: 0, bad: 0 }; // Placeholder
}

function calculateInnovationMetrics(entries) {
  return { total: 0, successful: 0, failed: 0 }; // Placeholder
}

function calculateBusinessTrends(entries) {
  return { revenue: 'stable', customers: 'stable', productivity: 'stable' }; // Placeholder
}

module.exports = router;
