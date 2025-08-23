# Cincinnatus Life Tracker - Express Backend

A comprehensive Express.js backend for tracking your life with prompts to follow, designed specifically for business professionals and life optimization enthusiasts.

## 🚀 Features

- **Life Tracking**: Daily entries with mood, energy, productivity, and stress tracking
- **Prompt System**: Curated prompts for business, personal growth, health, and more
- **Goal Management**: Set and track short-term, medium-term, and long-term goals
- **Business Analytics**: Track revenue, customers, projects, and business metrics
- **Personal Analytics**: Monitor health, relationships, learning, and personal growth
- **User Management**: Secure authentication with JWT and role-based access control
- **Comprehensive API**: RESTful API with validation, pagination, and error handling

## 🏗️ Architecture

- **Framework**: Express.js with Node.js
- **Database**: MongoDB with Mongoose ODM
- **Authentication**: JWT with bcrypt password hashing
- **Validation**: Express-validator for input validation
- **Security**: Helmet, CORS, rate limiting, and security middleware
- **File Structure**: Modular architecture with separate routes, models, and middleware

## 📋 Prerequisites

- Node.js (v16 or higher)
- MongoDB (local or cloud instance)
- npm or yarn package manager

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd cincinnatus-server
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment Setup**
   ```bash
   # Copy environment file
   cp env.example .env
   
   # Edit .env with your configuration
   nano .env
   ```

4. **Database Setup**
   - Ensure MongoDB is running
   - Update `MONGODB_URI` in your `.env` file
   - The database will be created automatically on first run

5. **Start the server**
   ```bash
   # Development mode with auto-reload
   npm run dev
   
   # Production mode
   npm start
   ```

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

```env
# Server Configuration
PORT=3000
NODE_ENV=development

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/cincinnatus-life-tracker

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-here
JWT_EXPIRES_IN=7d

# Security Configuration
BCRYPT_ROUNDS=12
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# File Upload Configuration
MAX_FILE_SIZE=5242880
UPLOAD_PATH=./uploads
```

## 🗄️ Database Models

### User
- Personal information and business details
- Preferences and settings
- Statistics and progress tracking
- Role-based access control

### Prompt
- Life tracking prompts with categories
- Difficulty levels and business relevance
- Usage tracking and ratings
- Resources and examples

### LifeEntry
- Daily life tracking entries
- Mood, energy, and productivity metrics
- Business and personal metrics
- Insights, actions, and reflections

### Goal
- Short-term, medium-term, and long-term goals
- Progress tracking and milestones
- Business and personal impact metrics
- Action items and resources

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user profile
- `PUT /api/auth/profile` - Update user profile
- `PUT /api/auth/change-password` - Change password
- `POST /api/auth/logout` - Logout user

### Users
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile
- `GET /api/users/stats` - Get user statistics
- `GET /api/users` - Get all users (admin only)
- `GET /api/users/:id` - Get user by ID (admin only)
- `PUT /api/users/:id` - Update user (admin only)
- `DELETE /api/users/:id` - Delete user (admin only)

### Prompts
- `GET /api/prompts` - Get all prompts with filtering
- `GET /api/prompts/:id` - Get single prompt
- `POST /api/prompts` - Create new prompt
- `PUT /api/prompts/:id` - Update prompt
- `DELETE /api/prompts/:id` - Delete prompt
- `POST /api/prompts/:id/rate` - Rate a prompt
- `GET /api/prompts/featured` - Get featured prompts
- `GET /api/prompts/business/:relevance` - Get prompts by business relevance

### Life Entries
- `GET /api/life-entries` - Get all life entries
- `GET /api/life-entries/:id` - Get single life entry
- `POST /api/life-entries` - Create new life entry
- `PUT /api/life-entries/:id` - Update life entry
- `DELETE /api/life-entries/:id` - Delete life entry
- `POST /api/life-entries/:id/insights` - Add insight
- `POST /api/life-entries/:id/actions` - Add action
- `PUT /api/life-entries/:id/actions/:actionId/complete` - Complete action
- `GET /api/life-entries/stats/overview` - Get user statistics
- `GET /api/life-entries/mood/:mood` - Get entries by mood

### Goals
- `GET /api/goals` - Get all goals
- `GET /api/goals/:id` - Get single goal
- `POST /api/goals` - Create new goal
- `PUT /api/goals/:id` - Update goal
- `DELETE /api/goals/:id` - Delete goal
- `PUT /api/goals/:id/progress` - Update goal progress
- `GET /api/goals/overdue` - Get overdue goals
- `GET /api/goals/category/:category` - Get goals by category

### Analytics
- `GET /api/analytics/overview` - Get analytics overview
- `GET /api/analytics/mood` - Get mood analytics
- `GET /api/analytics/business` - Get business analytics

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## 📊 Data Structure Examples

### Creating a Life Entry
```json
{
  "promptId": "60d21b4667d0d8992e610c85",
  "title": "Today's Reflection",
  "content": "Today was productive. I completed three major tasks and felt energized throughout the day.",
  "mood": "great",
  "energy": 8,
  "productivity": 9,
  "stress": 3,
  "satisfaction": 8,
  "insights": ["Working in focused blocks increases productivity"],
  "actions": [
    {
      "title": "Schedule focused work blocks",
      "description": "Block 2-hour periods for deep work",
      "priority": "high"
    }
  ],
  "businessMetrics": {
    "revenue": 5000,
    "customers": 3,
    "projects": 2
  },
  "personalMetrics": {
    "exercise": 45,
    "sleep": 7.5,
    "meditation": 20
  }
}
```

### Creating a Goal
```json
{
  "title": "Increase Monthly Revenue by 25%",
  "description": "Focus on expanding customer base and improving conversion rates",
  "category": "business",
  "type": "medium-term",
  "targetDate": "2024-06-30T00:00:00.000Z",
  "priority": "high",
  "businessImpact": {
    "revenue": 75,
    "customers": 60,
    "efficiency": 80
  }
}
```

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## 🚀 Deployment

### Production Build
```bash
# Install production dependencies only
npm ci --only=production

# Set NODE_ENV to production
export NODE_ENV=production

# Start the server
npm start
```

### Environment Variables for Production
- Set `NODE_ENV=production`
- Use a strong `JWT_SECRET`
- Configure production MongoDB URI
- Set appropriate rate limiting values
- Configure CORS origins for production domains

## 📈 Monitoring and Logging

The application includes:
- Request logging with Morgan
- Error logging and handling
- Health check endpoint at `/health`
- Rate limiting and security headers

## 🔧 Development

### Project Structure
```
cincinnatus-server/
├── config/          # Database and configuration
├── middleware/      # Authentication and validation
├── models/          # Mongoose models
├── routes/          # API route handlers
├── server.js        # Main server file
├── package.json     # Dependencies and scripts
└── README.md        # This file
```

### Adding New Features
1. Create model in `models/` directory
2. Add routes in `routes/` directory
3. Update `server.js` to include new routes
4. Add validation and error handling
5. Update documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation
- Review the code examples

## 🔮 Roadmap

- [ ] Habit tracking system
- [ ] Advanced analytics and reporting
- [ ] Mobile app support
- [ ] Team collaboration features
- [ ] Integration with external services
- [ ] Advanced goal templates
- [ ] AI-powered insights and recommendations

---

**Built with ❤️ for life optimization and business success**
