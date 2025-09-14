import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const TickerContainer = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  min-height: 400px;
  display: flex;
  flex-direction: column;
  justify-content: center;
`;

const SymbolHeader = styled.div`
  text-align: center;
  margin-bottom: 30px;
`;

const SymbolName = styled.h2`
  font-size: 2.5rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 10px 0;
`;

const CurrentPrice = styled.div`
  font-size: 3rem;
  font-weight: 600;
  color: #059669;
  margin-bottom: 10px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const PriceChange = styled.div`
  font-size: 1.2rem;
  color: #6b7280;
  font-weight: 500;
`;

const ForecastsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 30px;
`;

const ForecastCard = styled(motion.div)`
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  border: 1px solid #e2e8f0;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  }
`;

const ForecastHorizon = styled.div`
  font-size: 1.1rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
`;

const ForecastPrice = styled.div`
  font-size: 1.8rem;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 8px;
`;

const ForecastConfidence = styled.div`
  font-size: 0.9rem;
  color: #6b7280;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 4px;
  background: #e5e7eb;
  border-radius: 2px;
  margin-top: 8px;
  overflow: hidden;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
  width: ${props => props.confidence * 100}%;
  transition: width 0.3s ease;
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  color: #6b7280;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #e5e7eb;
  border-top: 4px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 20px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  color: #6b7280;
  font-size: 1.1rem;
  padding: 40px 20px;
`;

const ErrorState = styled.div`
  text-align: center;
  color: #ef4444;
  font-size: 1.1rem;
  padding: 40px 20px;
`;

function TickerDisplay({ forecast, symbol, isLoading }) {
  if (isLoading) {
    return (
      <TickerContainer>
        <LoadingContainer>
          <LoadingSpinner />
          <div>Loading forecast for {symbol}...</div>
        </LoadingContainer>
      </TickerContainer>
    );
  }

  if (!forecast) {
    return (
      <TickerContainer>
        <EmptyState>
          <div>📊</div>
          <div>Select a symbol to view forecasts</div>
        </EmptyState>
      </TickerContainer>
    );
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const formatConfidence = (confidence) => {
    return `${Math.round(confidence * 100)}%`;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981';
    if (confidence >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <TickerContainer>
      <SymbolHeader>
        <SymbolName>{symbol}</SymbolName>
        <CurrentPrice>{formatPrice(forecast.current_price)}</CurrentPrice>
        <PriceChange>Current Price</PriceChange>
      </SymbolHeader>

      <ForecastsGrid>
        {Object.entries(forecast.forecasts).map(([horizon, data]) => (
          <ForecastCard
            key={horizon}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            whileHover={{ scale: 1.05 }}
          >
            <ForecastHorizon>{horizon}</ForecastHorizon>
            <ForecastPrice>{formatPrice(data.price)}</ForecastPrice>
            <ForecastConfidence>
              <span>Confidence: {formatConfidence(data.confidence)}</span>
            </ForecastConfidence>
            <ConfidenceBar>
              <ConfidenceFill confidence={data.confidence} />
            </ConfidenceBar>
          </ForecastCard>
        ))}
      </ForecastsGrid>
    </TickerContainer>
  );
}

export default TickerDisplay;