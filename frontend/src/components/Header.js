import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const HeaderContainer = styled.header`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 20px 30px;
  margin-bottom: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const HeaderContent = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
`;

const Title = styled.h1`
  color: white;
  font-size: 2rem;
  font-weight: 700;
  margin: 0;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
`;

const StatusContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: white;
  font-weight: 500;
`;

const StatusDot = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${props => props.isHealthy ? '#4ade80' : '#ef4444'};
  animation: ${props => props.isHealthy ? 'pulse 2s infinite' : 'none'};
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const RefreshButton = styled(motion.button)`
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  
  &:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-1px);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const LastUpdate = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.9rem;
  text-align: right;
`;

const LoadingSpinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

function Header({ isHealthy, lastUpdate, onRefresh, isLoading }) {
  const formatLastUpdate = (date) => {
    if (!date) return 'Never';
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <HeaderContainer>
      <HeaderContent>
        <Title>🚀 Crypto Price Forecast</Title>
        
        <StatusContainer>
          <StatusIndicator>
            <StatusDot isHealthy={isHealthy} />
            {isHealthy ? 'Model Ready' : 'Model Loading...'}
          </StatusIndicator>
          
          <RefreshButton
            onClick={onRefresh}
            disabled={isLoading || !isHealthy}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isLoading ? <LoadingSpinner /> : '🔄'}
            Refresh
          </RefreshButton>
          
          <LastUpdate>
            Last update: {formatLastUpdate(lastUpdate)}
          </LastUpdate>
        </StatusContainer>
      </HeaderContent>
    </HeaderContainer>
  );
}

export default Header;