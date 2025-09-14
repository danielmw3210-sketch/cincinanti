import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.error || 'An error occurred';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const fetchForecast = async (symbol, lookbackHours = 24) => {
  try {
    const response = await api.post('/forecast', {
      symbol,
      timeframe: '1m',
      lookback_hours: lookbackHours,
    });
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch forecast: ${error.message}`);
  }
};

export const fetchSymbols = async () => {
  try {
    const response = await api.get('/symbols');
    return response.data.symbols;
  } catch (error) {
    throw new Error(`Failed to fetch symbols: ${error.message}`);
  }
};

export const fetchHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch health status: ${error.message}`);
  }
};

export const fetchModelInfo = async () => {
  try {
    const response = await api.get('/model/info');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch model info: ${error.message}`);
  }
};

export const retrainModel = async () => {
  try {
    const response = await api.post('/retrain');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to retrain model: ${error.message}`);
  }
};

export const fetchDataSummary = async (symbol = 'BTC/USDT', hoursBack = 24) => {
  try {
    const response = await api.get('/data/summary', {
      params: { symbol, hours_back: hoursBack }
    });
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch data summary: ${error.message}`);
  }
};

export default api;