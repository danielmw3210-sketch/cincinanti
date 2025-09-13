#!/usr/bin/env python3
"""Test the AI Crypto Trader Dashboard."""

import requests
import time
import json

def test_dashboard():
    """Test the dashboard endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing AI Crypto Trader Dashboard...")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test dashboard endpoint
    try:
        response = requests.get(f"{base_url}/api/trading/dashboard", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Dashboard data endpoint working")
                portfolio = data['data']['portfolio']
                print(f"   Total Balance: ${portfolio['totalBalance']:,.2f}")
                print(f"   Available Balance: ${portfolio['availableBalance']:,.2f}")
                print(f"   Daily P&L: ${portfolio['dailyPnl']:,.2f}")
            else:
                print(f"❌ Dashboard data failed: {data}")
                return False
        else:
            print(f"❌ Dashboard endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard endpoint failed: {e}")
        return False
    
    # Test main page
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Main dashboard page accessible")
            if "AI Crypto Trader" in response.text:
                print("   ✅ Dashboard content found")
            else:
                print("   ⚠️  Dashboard content not found")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Main page failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Dashboard is working correctly.")
    print(f"🌐 Dashboard URL: {base_url}")
    print(f"📚 API Docs: {base_url}/docs")
    return True

if __name__ == "__main__":
    success = test_dashboard()
    exit(0 if success else 1)