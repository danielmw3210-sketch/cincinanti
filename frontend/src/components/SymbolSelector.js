import React from 'react';
import styled from 'styled-components';
import Select from 'react-select';

const SelectorContainer = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 25px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const SelectorTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const CustomSelect = styled(Select)`
  .react-select__control {
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    min-height: 48px;
    box-shadow: none;
    background: white;
    
    &:hover {
      border-color: #3b82f6;
    }
    
    &--is-focused {
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
  }
  
  .react-select__value-container {
    padding: 0 12px;
  }
  
  .react-select__input-container {
    color: #1f2937;
    font-weight: 500;
  }
  
  .react-select__single-value {
    color: #1f2937;
    font-weight: 600;
  }
  
  .react-select__placeholder {
    color: #9ca3af;
  }
  
  .react-select__menu {
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
    margin-top: 4px;
  }
  
  .react-select__option {
    padding: 12px 16px;
    color: #374151;
    font-weight: 500;
    
    &--is-focused {
      background: #f3f4f6;
      color: #1f2937;
    }
    
    &--is-selected {
      background: #3b82f6;
      color: white;
    }
  }
  
  .react-select__indicator-separator {
    background: #e5e7eb;
  }
  
  .react-select__dropdown-indicator {
    color: #6b7280;
    
    &:hover {
      color: #374151;
    }
  }
`;

const SymbolInfo = styled.div`
  margin-top: 20px;
  padding: 15px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
`;

const SymbolName = styled.div`
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 5px;
`;

const SymbolDescription = styled.div`
  font-size: 0.9rem;
  color: #6b7280;
  line-height: 1.4;
`;

const LoadingContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100px;
  color: #6b7280;
`;

const LoadingSpinner = styled.div`
  width: 20px;
  height: 20px;
  border: 2px solid #e5e7eb;
  border-top: 2px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 10px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const getSymbolDescription = (symbol) => {
  const descriptions = {
    'BTC/GBP': 'Bitcoin - The first and largest cryptocurrency',
    'ETH/GBP': 'Ethereum - Smart contract platform and cryptocurrency',
    'ADA/GBP': 'Cardano - Blockchain platform for smart contracts',
    'SOL/GBP': 'Solana - High-performance blockchain platform',
    'XRP/GBP': 'Ripple - Digital payment protocol',
    'DOT/GBP': 'Polkadot - Multi-chain blockchain platform',
    'LTC/GBP': 'Litecoin - Digital silver to Bitcoin\'s gold',
    'LINK/GBP': 'Chainlink - Decentralized oracle network',
    'AVAX/GBP': 'Avalanche - Smart contract platform',
    'BTC/USDT': 'Bitcoin - The first and largest cryptocurrency',
    'ETH/USDT': 'Ethereum - Smart contract platform and cryptocurrency',
    'BNB/USDT': 'Binance Coin - Native token of Binance exchange',
    'ADA/USDT': 'Cardano - Blockchain platform for smart contracts',
    'SOL/USDT': 'Solana - High-performance blockchain platform',
    'XRP/USDT': 'Ripple - Digital payment protocol',
    'DOT/USDT': 'Polkadot - Multi-chain blockchain platform',
    'DOGE/USDT': 'Dogecoin - Meme cryptocurrency',
    'AVAX/USDT': 'Avalanche - Smart contract platform',
    'MATIC/USDT': 'Polygon - Ethereum scaling solution'
  };
  
  return descriptions[symbol] || 'Cryptocurrency trading pair';
};

const getSymbolIcon = (symbol) => {
  const icons = {
    'BTC/GBP': '₿',
    'ETH/GBP': 'Ξ',
    'ADA/GBP': '🔵',
    'SOL/GBP': '🟣',
    'XRP/GBP': '🔵',
    'DOT/GBP': '🟣',
    'LTC/GBP': 'Ł',
    'LINK/GBP': '🔗',
    'AVAX/GBP': '🔺',
    'BTC/USDT': '₿',
    'ETH/USDT': 'Ξ',
    'BNB/USDT': '🟡',
    'ADA/USDT': '🔵',
    'SOL/USDT': '🟣',
    'XRP/USDT': '🔵',
    'DOT/USDT': '🟣',
    'DOGE/USDT': '🐕',
    'AVAX/USDT': '🔺',
    'MATIC/USDT': '🔷'
  };
  
  return icons[symbol] || '💰';
};

function SymbolSelector({ symbols, selectedSymbol, onSymbolChange, disabled }) {
  const options = symbols.map(symbol => ({
    value: symbol,
    label: `${getSymbolIcon(symbol)} ${symbol}`,
    symbol: symbol
  }));

  const selectedOption = options.find(option => option.value === selectedSymbol);

  const customStyles = {
    control: (provided) => ({
      ...provided,
      border: '2px solid #e5e7eb',
      borderRadius: '8px',
      minHeight: '48px',
      boxShadow: 'none',
      '&:hover': {
        borderColor: '#3b82f6'
      }
    }),
    option: (provided, state) => ({
      ...provided,
      backgroundColor: state.isSelected 
        ? '#3b82f6' 
        : state.isFocused 
          ? '#f3f4f6' 
          : 'white',
      color: state.isSelected ? 'white' : '#374151',
      fontWeight: '500',
      padding: '12px 16px'
    }),
    singleValue: (provided) => ({
      ...provided,
      color: '#1f2937',
      fontWeight: '600'
    })
  };

  if (symbols.length === 0) {
    return (
      <SelectorContainer>
        <SelectorTitle>🔍 Symbol Selector</SelectorTitle>
        <LoadingContainer>
          <LoadingSpinner />
          Loading symbols...
        </LoadingContainer>
      </SelectorContainer>
    );
  }

  return (
    <SelectorContainer>
      <SelectorTitle>🔍 Symbol Selector</SelectorTitle>
      
      <CustomSelect
        value={selectedOption}
        onChange={(option) => onSymbolChange(option.value)}
        options={options}
        isDisabled={disabled}
        isSearchable={true}
        placeholder="Select a cryptocurrency..."
        classNamePrefix="react-select"
        styles={customStyles}
      />
      
      {selectedSymbol && (
        <SymbolInfo>
          <SymbolName>
            {getSymbolIcon(selectedSymbol)} {selectedSymbol}
          </SymbolName>
          <SymbolDescription>
            {getSymbolDescription(selectedSymbol)}
          </SymbolDescription>
        </SymbolInfo>
      )}
    </SelectorContainer>
  );
}

export default SymbolSelector;