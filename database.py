"""Database models and operations for trading system."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
import json
from enum import Enum

# Database configuration
DATABASE_URL = "postgresql://trader:password@localhost:5432/trading_db"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class OrderType(Enum):
    """Order type enumeration."""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"

class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"

class TradingPlatform(Enum):
    """Trading platform enumeration."""
    MT5 = "mt5"
    IB = "interactive_brokers"
    CCXT = "ccxt"
    KRAKEN = "kraken"

class Order(Base):
    """Order model for database storage."""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket = Column(Integer, unique=True, index=True)
    platform = Column(SQLEnum(TradingPlatform), nullable=False)
    symbol = Column(String(50), nullable=False, index=True)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    volume = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    comment = Column(Text, nullable=True)
    magic = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)

class Position(Base):
    """Position model for database storage."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket = Column(Integer, unique=True, index=True)
    platform = Column(SQLEnum(TradingPlatform), nullable=False)
    symbol = Column(String(50), nullable=False, index=True)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    price_current = Column(Float, nullable=True)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    profit = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    status = Column(SQLEnum(PositionStatus), nullable=False, default=PositionStatus.OPEN)
    comment = Column(Text, nullable=True)
    magic = Column(Integer, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB, nullable=True)

class Trade(Base):
    """Trade execution model for database storage."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, nullable=True)
    position_id = Column(Integer, nullable=True)
    platform = Column(SQLEnum(TradingPlatform), nullable=False)
    symbol = Column(String(50), nullable=False, index=True)
    trade_type = Column(SQLEnum(OrderType), nullable=False)
    volume = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    profit = Column(Float, default=0.0)
    executed_at = Column(DateTime, default=datetime.utcnow)
    comment = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)

class RiskEvent(Base):
    """Risk management events model."""
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    message = Column(Text, nullable=False)
    symbol = Column(String(50), nullable=True, index=True)
    platform = Column(SQLEnum(TradingPlatform), nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    resolved = Column(Boolean, default=False)

class SystemLog(Base):
    """System logging model."""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True)
    function = Column(String(100), nullable=True)
    line_number = Column(Integer, nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database manager for trading operations."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database manager."""
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def log_order(self, order_data: Dict[str, Any]) -> Optional[int]:
        """Log order to database."""
        try:
            with self.get_session() as session:
                order = Order(
                    ticket=order_data.get('ticket'),
                    platform=TradingPlatform(order_data.get('platform', 'mt5')),
                    symbol=order_data.get('symbol'),
                    order_type=OrderType(order_data.get('order_type')),
                    volume=order_data.get('volume'),
                    price=order_data.get('price'),
                    sl=order_data.get('sl'),
                    tp=order_data.get('tp'),
                    status=OrderStatus(order_data.get('status', 'pending')),
                    comment=order_data.get('comment'),
                    magic=order_data.get('magic'),
                    metadata=order_data.get('metadata')
                )
                
                session.add(order)
                session.commit()
                session.refresh(order)
                
                logger.info(f"Order logged to database: {order.id}")
                return order.id
                
        except Exception as e:
            logger.error(f"Error logging order: {e}")
            return None
    
    def log_position(self, position_data: Dict[str, Any]) -> Optional[int]:
        """Log position to database."""
        try:
            with self.get_session() as session:
                position = Position(
                    ticket=position_data.get('ticket'),
                    platform=TradingPlatform(position_data.get('platform', 'mt5')),
                    symbol=position_data.get('symbol'),
                    order_type=OrderType(position_data.get('order_type')),
                    volume=position_data.get('volume'),
                    price_open=position_data.get('price_open'),
                    price_current=position_data.get('price_current'),
                    sl=position_data.get('sl'),
                    tp=position_data.get('tp'),
                    profit=position_data.get('profit', 0.0),
                    swap=position_data.get('swap', 0.0),
                    status=PositionStatus(position_data.get('status', 'open')),
                    comment=position_data.get('comment'),
                    magic=position_data.get('magic'),
                    metadata=position_data.get('metadata')
                )
                
                session.add(position)
                session.commit()
                session.refresh(position)
                
                logger.info(f"Position logged to database: {position.id}")
                return position.id
                
        except Exception as e:
            logger.error(f"Error logging position: {e}")
            return None
    
    def log_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Log trade execution to database."""
        try:
            with self.get_session() as session:
                trade = Trade(
                    order_id=trade_data.get('order_id'),
                    position_id=trade_data.get('position_id'),
                    platform=TradingPlatform(trade_data.get('platform', 'mt5')),
                    symbol=trade_data.get('symbol'),
                    trade_type=OrderType(trade_data.get('trade_type')),
                    volume=trade_data.get('volume'),
                    price=trade_data.get('price'),
                    commission=trade_data.get('commission', 0.0),
                    swap=trade_data.get('swap', 0.0),
                    profit=trade_data.get('profit', 0.0),
                    comment=trade_data.get('comment'),
                    metadata=trade_data.get('metadata')
                )
                
                session.add(trade)
                session.commit()
                session.refresh(trade)
                
                logger.info(f"Trade logged to database: {trade.id}")
                return trade.id
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return None
    
    def log_risk_event(self, event_data: Dict[str, Any]) -> Optional[int]:
        """Log risk management event to database."""
        try:
            with self.get_session() as session:
                risk_event = RiskEvent(
                    event_type=event_data.get('event_type'),
                    severity=event_data.get('severity', 'medium'),
                    message=event_data.get('message'),
                    symbol=event_data.get('symbol'),
                    platform=TradingPlatform(event_data.get('platform')) if event_data.get('platform') else None,
                    metadata=event_data.get('metadata')
                )
                
                session.add(risk_event)
                session.commit()
                session.refresh(risk_event)
                
                logger.info(f"Risk event logged to database: {risk_event.id}")
                return risk_event.id
                
        except Exception as e:
            logger.error(f"Error logging risk event: {e}")
            return None
    
    def log_system_event(self, level: str, message: str, module: str = None, 
                        function: str = None, line_number: int = None, 
                        metadata: Dict = None) -> Optional[int]:
        """Log system event to database."""
        try:
            with self.get_session() as session:
                system_log = SystemLog(
                    level=level,
                    message=message,
                    module=module,
                    function=function,
                    line_number=line_number,
                    metadata=metadata
                )
                
                session.add(system_log)
                session.commit()
                session.refresh(system_log)
                
                return system_log.id
                
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            return None
    
    def update_order_status(self, ticket: int, status: OrderStatus, 
                          rejection_reason: str = None) -> bool:
        """Update order status."""
        try:
            with self.get_session() as session:
                order = session.query(Order).filter(Order.ticket == ticket).first()
                if not order:
                    logger.warning(f"Order {ticket} not found")
                    return False
                
                order.status = status
                order.updated_at = datetime.utcnow()
                
                if status == OrderStatus.FILLED:
                    order.filled_at = datetime.utcnow()
                elif status == OrderStatus.CANCELLED:
                    order.cancelled_at = datetime.utcnow()
                elif status == OrderStatus.REJECTED:
                    order.rejection_reason = rejection_reason
                
                session.commit()
                logger.info(f"Order {ticket} status updated to {status}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def update_position(self, ticket: int, **kwargs) -> bool:
        """Update position data."""
        try:
            with self.get_session() as session:
                position = session.query(Position).filter(Position.ticket == ticket).first()
                if not position:
                    logger.warning(f"Position {ticket} not found")
                    return False
                
                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                position.updated_at = datetime.utcnow()
                session.commit()
                
                logger.info(f"Position {ticket} updated")
                return True
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return False
    
    def get_orders(self, platform: TradingPlatform = None, 
                  status: OrderStatus = None, limit: int = 100) -> List[Dict]:
        """Get orders from database."""
        try:
            with self.get_session() as session:
                query = session.query(Order)
                
                if platform:
                    query = query.filter(Order.platform == platform)
                if status:
                    query = query.filter(Order.status == status)
                
                orders = query.order_by(Order.created_at.desc()).limit(limit).all()
                
                return [{
                    'id': order.id,
                    'ticket': order.ticket,
                    'platform': order.platform.value,
                    'symbol': order.symbol,
                    'order_type': order.order_type.value,
                    'volume': order.volume,
                    'price': order.price,
                    'sl': order.sl,
                    'tp': order.tp,
                    'status': order.status.value,
                    'comment': order.comment,
                    'magic': order.magic,
                    'created_at': order.created_at.isoformat(),
                    'updated_at': order.updated_at.isoformat(),
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'cancelled_at': order.cancelled_at.isoformat() if order.cancelled_at else None,
                    'rejection_reason': order.rejection_reason,
                    'metadata': order.metadata
                } for order in orders]
                
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_positions(self, platform: TradingPlatform = None, 
                     status: PositionStatus = None, limit: int = 100) -> List[Dict]:
        """Get positions from database."""
        try:
            with self.get_session() as session:
                query = session.query(Position)
                
                if platform:
                    query = query.filter(Position.platform == platform)
                if status:
                    query = query.filter(Position.status == status)
                
                positions = query.order_by(Position.opened_at.desc()).limit(limit).all()
                
                return [{
                    'id': position.id,
                    'ticket': position.ticket,
                    'platform': position.platform.value,
                    'symbol': position.symbol,
                    'order_type': position.order_type.value,
                    'volume': position.volume,
                    'price_open': position.price_open,
                    'price_current': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'status': position.status.value,
                    'comment': position.comment,
                    'magic': position.magic,
                    'opened_at': position.opened_at.isoformat(),
                    'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                    'updated_at': position.updated_at.isoformat(),
                    'metadata': position.metadata
                } for position in positions]
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_risk_events(self, severity: str = None, resolved: bool = None, 
                       limit: int = 100) -> List[Dict]:
        """Get risk events from database."""
        try:
            with self.get_session() as session:
                query = session.query(RiskEvent)
                
                if severity:
                    query = query.filter(RiskEvent.severity == severity)
                if resolved is not None:
                    query = query.filter(RiskEvent.resolved == resolved)
                
                events = query.order_by(RiskEvent.created_at.desc()).limit(limit).all()
                
                return [{
                    'id': event.id,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'message': event.message,
                    'symbol': event.symbol,
                    'platform': event.platform.value if event.platform else None,
                    'metadata': event.metadata,
                    'created_at': event.created_at.isoformat(),
                    'resolved_at': event.resolved_at.isoformat() if event.resolved_at else None,
                    'resolved': event.resolved
                } for event in events]
                
        except Exception as e:
            logger.error(f"Error getting risk events: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager()