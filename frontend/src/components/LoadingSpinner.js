import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const SpinnerContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 40px;
`;

const Spinner = styled.div`
  width: ${props => props.size || 40}px;
  height: ${props => props.size || 40}px;
  border: ${props => props.thickness || 4}px solid #e5e7eb;
  border-top: ${props => props.thickness || 4}px solid #3b82f6;
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const LoadingText = styled.div`
  color: #6b7280;
  font-size: ${props => props.fontSize || '1rem'};
  font-weight: 500;
  text-align: center;
`;

function LoadingSpinner({ 
  size = 40, 
  thickness = 4, 
  text = 'Loading...', 
  fontSize = '1rem' 
}) {
  return (
    <SpinnerContainer>
      <Spinner size={size} thickness={thickness} />
      {text && <LoadingText fontSize={fontSize}>{text}</LoadingText>}
    </SpinnerContainer>
  );
}

export default LoadingSpinner;