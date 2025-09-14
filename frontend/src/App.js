import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Header from './components/Header';
import TickerDisplay from './components/TickerDisplay';
import ForecastPanel from './components/ForecastPanel';
import SymbolSelector from './components/SymbolSelector';
import LoadingSpinner from './components/LoadingSpinner';
import { fetchForecast, fetchSymbols, fetchHealth } from './services/api';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
`;

const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 20px;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 15px;
  }
`;

const LeftPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const RightPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

function App() {
  const [forecasts, setForecasts] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [availableSymbols, setAvailableSymbols] = useState(['BTC/USDT']);
  const [isLoading, setIsLoading] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch available symbols on component mount
  useEffect(() => {
    const loadSymbols = async () => {
      try {
        const symbols = await fetchSymbols();
        setAvailableSymbols(symbols);
      } catch (error) {
        console.error('Failed to load symbols:', error);
        toast.error('Failed to load available symbols');
      }
    };

    loadSymbols();
  }, []);

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await fetchHealth();
        setIsHealthy(health.model_loaded);
        if (!health.model_loaded) {
          toast.warning('Model is not loaded. Some features may not work.');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        toast.error('Failed to connect to the API');
      }
    };

    checkHealth();
  }, []);

  // Auto-refresh forecasts every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (selectedSymbol && isHealthy) {
        handleForecast(selectedSymbol);
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [selectedSymbol, isHealthy]);

  const handleForecast = async (symbol = selectedSymbol) => {
    if (!symbol) return;

    setIsLoading(true);
    try {
      const forecastData = await fetchForecast(symbol);
      setForecasts(prev => ({
        ...prev,
        [symbol]: forecastData
      }));
      setLastUpdate(new Date());
      toast.success(`Forecast updated for ${symbol}`);
    } catch (error) {
      console.error('Forecast failed:', error);
      toast.error(`Failed to get forecast for ${symbol}: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
    if (isHealthy) {
      handleForecast(symbol);
    }
  };

  const currentForecast = forecasts[selectedSymbol];

  return (
    <AppContainer>
      <Header 
        isHealthy={isHealthy}
        lastUpdate={lastUpdate}
        onRefresh={() => handleForecast()}
        isLoading={isLoading}
      />
      
      <MainContent>
        <LeftPanel>
          <TickerDisplay 
            forecast={currentForecast}
            symbol={selectedSymbol}
            isLoading={isLoading}
          />
        </LeftPanel>
        
        <RightPanel>
          <SymbolSelector
            symbols={availableSymbols}
            selectedSymbol={selectedSymbol}
            onSymbolChange={handleSymbolChange}
            disabled={!isHealthy}
          />
          
          <ForecastPanel
            forecast={currentForecast}
            symbol={selectedSymbol}
            isLoading={isLoading}
          />
        </RightPanel>
      </MainContent>

      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </AppContainer>
  );
}

export default App;