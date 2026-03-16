import requests
import json
import os


ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    "Content-Type":        "application/json"
}


def getAccountBalance() -> float:
    response = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS)
    return float(response.json()['portfolio_value'])


def getCalendarSpreadMid(front_symbol: str, back_symbol: str) -> float | None:
    def getOptionMid(symbol: str) -> float | None:
        response = requests.get(
            f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}",
            headers=HEADERS
        )
        data  = response.json()
        quote = data.get('snapshot', {}).get('latestQuote', {})
        bid   = quote.get('bp', 0)
        ask   = quote.get('ap', 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return None

    front_mid = getOptionMid(front_symbol)
    back_mid  = getOptionMid(back_symbol)

    if front_mid is None or back_mid is None:
        return None

    return round(back_mid - front_mid, 2)


def getContractCount(portfolio_value: float, net_debit: float, kelly: float = 0.065) -> int:
    max_risk          = portfolio_value * kelly
    cost_per_contract = net_debit * 100
    contracts         = int(max_risk / cost_per_contract)
    return max(1, contracts)


def placeCalendarSpread(front_symbol: str, back_symbol: str, quantity: int, net_debit: float) -> dict:
    order = {
        "order_class":   "mleg",
        "qty":           str(quantity),
        "type":          "limit",
        "limit_price":   str(round(net_debit, 2)),
        "time_in_force": "day",
        "legs": [
            {
                "symbol":          front_symbol,
                "ratio_qty":       "1",
                "side":            "sell",
                "position_intent": "sell_to_open"
            },
            {
                "symbol":          back_symbol,
                "ratio_qty":       "1",
                "side":            "buy",
                "position_intent": "buy_to_open"
            }
        ]
    }

    response = requests.post(f"{ALPACA_BASE_URL}/v2/orders", json=order, headers=HEADERS)
    return response.json()


def executeTrades(legs: dict) -> None:
    if not legs:
        print("No legs to execute.")
        return

    portfolio_value = getAccountBalance()
    print(f"\nPortfolio value:           ${portfolio_value:,.2f}")
    print(f"Max risk per trade (6.5%): ${portfolio_value * 0.065:,.2f}\n")

    for ticker, leg in legs.items():
        if leg.get('error'):
            print(f"{ticker:6} | ERROR: {leg['error']}")
            continue

        front_symbol = leg['front_symbol']
        back_symbol  = leg['back_symbol']

        net_debit = getCalendarSpreadMid(front_symbol, back_symbol)
        if net_debit is None or net_debit <= 0:
            print(f"{ticker:6} | ERROR: Could not determine net debit")
            continue

        contracts  = getContractCount(portfolio_value, net_debit)
        total_cost = contracts * net_debit * 100

        print(f"{ticker:6} | Debit: ${net_debit:.2f} | Contracts: {contracts} | Total: ${total_cost:.2f}")
        print(f"{'':6}   Sell: {front_symbol}")
        print(f"{'':6}   Buy:  {back_symbol}")

        result = placeCalendarSpread(front_symbol, back_symbol, contracts, net_debit)

        if result.get('id'):
            print(f"{'':6}   Order placed ✓ | ID: {result['id']}\n")
        else:
            print(f"{'':6}   Order failed  ✗ | {result}\n")


if __name__ == "__main__":
    try:
        with open("legs.json", "r") as f:
            legs = json.load(f)
        print(f"Loaded {len(legs)} leg(s) from legs.json\n")
    except FileNotFoundError:
        print("legs.json not found — run scan.py first")
        exit()

    executeTrades(legs)
