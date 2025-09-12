#!/usr/bin/env python3
"""Convenience script to run the AI Crypto Trader with different modes."""

import sys
import os
import argparse
import json
from datetime import datetime
from main import AICryptoTrader

def main():
    parser = argparse.ArgumentParser(description="AI Crypto Trader for Kraken Pro")
    parser.add_argument("command", choices=["start", "analyze", "status", "trade", "backtest"], 
                       help="Command to execute")
    parser.add_argument("--pair", "-p", default="BTC/USD", 
                       help="Trading pair (default: BTC/USD)")
    parser.add_argument("--action", "-a", choices=["buy", "sell"], 
                       help="Trade action (for trade command)")
    parser.add_argument("--size", "-s", type=float, 
                       help="Trade size (for trade command)")
    parser.add_argument("--output", "-o", 
                       help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize trader
    trader = AICryptoTrader()
    
    try:
        if args.command == "start":
            print("🚀 Starting AI Crypto Trader...")
            print("Press Ctrl+C to stop")
            trader.start()
            
            # Keep running until interrupted
            import time
            try:
                while trader.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping trader...")
                trader.stop()
        
        elif args.command == "analyze":
            print(f"📊 Analyzing {args.pair}...")
            result = trader.run_single_analysis(args.pair)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Results saved to {args.output}")
            else:
                print(json.dumps(result, indent=2, default=str))
        
        elif args.command == "status":
            print("📈 Getting system status...")
            status = trader.get_status()
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(status, f, indent=2, default=str)
                print(f"Status saved to {args.output}")
            else:
                print(json.dumps(status, indent=2, default=str))
        
        elif args.command == "trade":
            if not args.action:
                print("❌ Error: --action is required for trade command")
                sys.exit(1)
            
            print(f"💰 Executing {args.action} trade for {args.pair}...")
            result = trader.execute_manual_trade(args.pair, args.action, args.size)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Trade result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2, default=str))
        
        elif args.command == "backtest":
            print(f"🧪 Running backtest for {args.pair}...")
            # This would need to be implemented
            print("Backtesting feature coming soon!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()