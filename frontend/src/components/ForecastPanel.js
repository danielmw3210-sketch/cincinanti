import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const PanelContainer = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 25px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const PanelTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const InfoGrid = styled.div`
  display: grid;
  gap: 15px;
`;

const InfoItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
`;

const InfoLabel = styled.span`
  font-weight: 500;
  color: #374151;
`;

const InfoValue = styled.span`
  font-weight: 600;
  color: #1f2937;
`;

const ModelInfo = styled.div`
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e5e7eb;
`;

const ModelType = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
`;

const ModelBadge = styled.span`
  background: ${props => props.type === 'xgboost' ? '#10b981' : '#f59e0b'};
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
`;

const DataQuality = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
`;

const QualityIndicator = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => {
    switch (props.quality) {
      case 'good': return '#10b981';
      case 'has_missing_values': return '#f59e0b';
      default: return '#ef4444';
    }
  }};
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #6b7280;
`;

const LoadingSpinner = styled.div`
  width: 30px;
  height: 30px;
  border: 3px solid #e5e7eb;
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 15px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  color: #6b7280;
  font-size: 1rem;
  padding: 40px 20px;
`;

function ForecastPanel({ forecast, symbol, isLoading }) {
  if (isLoading) {
    return (
      <PanelContainer>
        <LoadingContainer>
          <LoadingSpinner />
          <div>Loading forecast details...</div>
        </LoadingContainer>
      </PanelContainer>
    );
  }

  if (!forecast) {
    return (
      <PanelContainer>
        <EmptyState>
          <div>📈</div>
          <div>No forecast data available</div>
        </EmptyState>
      </PanelContainer>
    );
  }

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getPriceChange = (currentPrice, forecastPrice) => {
    const change = forecastPrice - currentPrice;
    const changePercent = (change / currentPrice) * 100;
    return {
      value: change,
      percent: changePercent,
      isPositive: change >= 0
    };
  };

  return (
    <PanelContainer>
      <PanelTitle>
        📊 Forecast Details
      </PanelTitle>

      <InfoGrid>
        <InfoItem>
          <InfoLabel>Symbol</InfoLabel>
          <InfoValue>{forecast.symbol}</InfoValue>
        </InfoItem>

        <InfoItem>
          <InfoLabel>Current Price</InfoLabel>
          <InfoValue>
            ${forecast.current_price.toLocaleString('en-US', {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2
            })}
          </InfoValue>
        </InfoItem>

        <InfoItem>
          <InfoLabel>Prediction Time</InfoLabel>
          <InfoValue>{formatTimestamp(forecast.timestamp)}</InfoValue>
        </InfoItem>

        <InfoItem>
          <InfoLabel>Model Type</InfoLabel>
          <InfoValue>
            <ModelBadge type={forecast.model_type}>
              {forecast.model_type}
            </ModelBadge>
          </InfoValue>
        </InfoItem>

        <InfoItem>
          <InfoLabel>Data Quality</InfoLabel>
          <InfoValue>
            <DataQuality>
              <QualityIndicator quality={forecast.data_quality} />
              {forecast.data_quality.replace('_', ' ').toUpperCase()}
            </DataQuality>
          </InfoValue>
        </InfoItem>
      </InfoGrid>

      <ModelInfo>
        <h4 style={{ margin: '0 0 15px 0', color: '#374151', fontSize: '1.1rem' }}>
          Price Predictions
        </h4>
        
        {Object.entries(forecast.forecasts).map(([horizon, data]) => {
          const priceChange = getPriceChange(forecast.current_price, data.price);
          
          return (
            <motion.div
              key={horizon}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px 0',
                borderBottom: '1px solid #f3f4f6'
              }}
            >
              <div>
                <div style={{ fontWeight: '600', color: '#1f2937' }}>
                  {horizon}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>
                  Confidence: {Math.round(data.confidence * 100)}%
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontWeight: '600', color: '#1f2937' }}>
                  ${data.price.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                  })}
                </div>
                <div style={{ 
                  fontSize: '0.9rem', 
                  color: priceChange.isPositive ? '#10b981' : '#ef4444',
                  fontWeight: '500'
                }}>
                  {priceChange.isPositive ? '+' : ''}
                  {priceChange.percent.toFixed(2)}%
                </div>
              </div>
            </motion.div>
          );
        })}
      </ModelInfo>
    </PanelContainer>
  );
}

export default ForecastPanel;