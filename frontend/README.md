# Cincinnatus Life Tracker - Frontend

A modern, modular React frontend for the Cincinnatus Life Tracker application. Built with React 18, Tailwind CSS, and modern tooling for a responsive and intuitive user experience.

## 🚀 Features

- **Modern React Architecture**: Built with React 18, hooks, and functional components
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Modular Components**: Reusable, well-structured components
- **State Management**: Zustand for lightweight state management
- **Data Fetching**: React Query for efficient API communication
- **Form Handling**: React Hook Form for form validation and management
- **Charts & Analytics**: Recharts for data visualization
- **Authentication**: JWT-based authentication with protected routes
- **Real-time Updates**: Optimistic updates and background refetching

## 🛠️ Tech Stack

- **Frontend Framework**: React 18
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: React Query (TanStack Query)
- **Forms**: React Hook Form
- **Routing**: React Router DOM
- **Charts**: Recharts
- **Icons**: Lucide React
- **Notifications**: React Hot Toast
- **Build Tool**: Create React App
- **Package Manager**: npm

## 📁 Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── common/           # Reusable components
│   │   │   ├── LoadingSpinner.js
│   │   │   ├── EmptyState.js
│   │   │   └── Modal.js
│   │   └── layout/           # Layout components
│   │       ├── Layout.js
│   │       ├── Sidebar.js
│   │       └── Header.js
│   ├── pages/                # Page components
│   │   ├── Dashboard.js
│   │   ├── Prompts.js
│   │   ├── LifeEntries.js
│   │   ├── Goals.js
│   │   ├── Analytics.js
│   │   ├── Profile.js
│   │   ├── Login.js
│   │   └── Register.js
│   ├── stores/               # State management
│   │   └── authStore.js
│   ├── services/             # API services
│   │   └── api.js
│   ├── App.js               # Main app component
│   ├── index.js             # Entry point
│   └── index.css            # Global styles
├── package.json
├── tailwind.config.js
└── postcss.config.js
```

## 🚀 Getting Started

### Prerequisites

- Node.js 16+ 
- npm or yarn
- Backend server running (see backend README)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cincinnatus-server/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment Configuration**
   Create a `.env` file in the frontend directory:
   ```env
   REACT_APP_API_URL=http://localhost:3000
   ```

4. **Start the development server**
   ```bash
   npm start
   ```

   The app will open at [http://localhost:3001](http://localhost:3001)

## 🏗️ Component Architecture

### Core Components

#### Layout Components
- **Layout**: Main application wrapper with sidebar and header
- **Sidebar**: Navigation sidebar with menu items
- **Header**: Top header with user profile and notifications

#### Common Components
- **LoadingSpinner**: Reusable loading indicator
- **EmptyState**: Empty state display for lists
- **Modal**: Reusable modal dialog component

#### Page Components
- **Dashboard**: Overview with stats and quick actions
- **Prompts**: Browse and filter available prompts
- **LifeEntries**: View and manage life tracking entries
- **Goals**: Track personal and business goals
- **Analytics**: Charts and insights from data
- **Profile**: User profile management
- **Login/Register**: Authentication forms

### State Management

The application uses Zustand for state management with the following stores:

- **authStore**: Authentication state, user data, and auth methods
- **Persistent storage**: JWT tokens are persisted in localStorage

### API Integration

- **Axios**: HTTP client with interceptors
- **React Query**: Data fetching, caching, and synchronization
- **Automatic token handling**: JWT tokens are automatically included in requests
- **Error handling**: Centralized error handling with toast notifications

## 🎨 Styling System

### Tailwind CSS Configuration

- **Custom color palette**: Primary, secondary, success, warning, danger colors
- **Component classes**: Pre-built component styles (buttons, cards, forms)
- **Responsive design**: Mobile-first approach with breakpoint utilities
- **Custom animations**: Fade-in, slide-up, and bounce animations

### Component Classes

```css
/* Buttons */
.btn, .btn-primary, .btn-secondary, .btn-success, .btn-warning, .btn-danger

/* Cards */
.card, .card-header, .card-title, .card-subtitle

/* Forms */
.form-input, .form-label, .form-error

/* Badges */
.badge, .badge-primary, .badge-secondary, .badge-success, .badge-warning, .badge-danger
```

## 📱 Responsive Design

The application is built with a mobile-first approach:

- **Mobile**: Single column layout with collapsible sidebar
- **Tablet**: Two-column grid layouts
- **Desktop**: Full sidebar with multi-column grids
- **Touch-friendly**: Optimized for touch interactions

## 🔐 Authentication Flow

1. **Login/Register**: User credentials are validated
2. **JWT Token**: Server returns JWT token on successful auth
3. **Token Storage**: Token is stored in localStorage
4. **Auto-include**: Token is automatically included in API requests
5. **Route Protection**: Protected routes check authentication status
6. **Token Refresh**: Automatic token validation on app initialization

## 📊 Data Visualization

### Charts Library
- **Recharts**: Modern charting library for React
- **Chart Types**: Line charts, bar charts, pie charts
- **Responsive**: Charts automatically resize for different screen sizes
- **Interactive**: Hover tooltips and click interactions

### Available Charts
- Mood trends over time
- Mood distribution pie chart
- Energy vs productivity correlation
- Business metrics bar charts

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

## 🚀 Building for Production

```bash
# Build the application
npm run build

# The build folder will contain the production-ready files
```

## 🔧 Development Scripts

```bash
npm start          # Start development server
npm run build      # Build for production
npm test           # Run tests
npm run eject      # Eject from Create React App (not recommended)
```

## 📦 Dependencies

### Core Dependencies
- **React**: UI library
- **React Router**: Client-side routing
- **React Query**: Data fetching and caching
- **Zustand**: State management
- **Tailwind CSS**: Utility-first CSS framework

### Development Dependencies
- **Create React App**: Build tool and development server
- **PostCSS**: CSS processing
- **Autoprefixer**: CSS vendor prefixing

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the documentation
- Review existing issues
- Create a new issue with detailed information

## 🚧 Roadmap

- [ ] Dark mode support
- [ ] Offline capabilities
- [ ] Push notifications
- [ ] Advanced analytics
- [ ] Mobile app (React Native)
- [ ] Export functionality
- [ ] Integration with external services
