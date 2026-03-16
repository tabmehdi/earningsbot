import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import requests


def filterDates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")


def yangZhang(price_data, window=30, trading_periods=252):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = (log_cc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol  = (log_oc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    return ((open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)).iloc[-1]


def buildIVCurve(days, ivs):
    days, ivs = np.array(days), np.array(ivs)
    idx = days.argsort()
    days, ivs = days[idx], ivs[idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    def term_spline(dte):
        if dte < days[0]:   return ivs[0]
        elif dte > days[-1]: return ivs[-1]
        else:                return float(spline(dte))
    return term_spline


def _emptyResult(error: str) -> dict:
    return {
        'recommendation': 'Avoid',
        'avg_volume':     False,
        'iv30_rv30':      False,
        'ts_slope_0_45':  False,
        'expected_move':  None,
        'error':          error
    }


def isTickerEligible(tickers) -> dict:
    """
    Accepts either a single ticker string, a list, or a dict of tickers.
    Returns a dict keyed by ticker with their eligibility results.
    """
    # normalize input to a list of tickers
    if isinstance(tickers, str):
        tickers = [tickers]
    elif isinstance(tickers, dict):
        tickers = list(tickers.keys())

    results = {}

    for ticker in tickers:
        try:
            ticker = ticker.strip().upper()
            stock  = yf.Ticker(ticker)

            if not stock.options:
                results[ticker] = _emptyResult(f"No options found for '{ticker}'")
                continue

            try:
                exp_dates = filterDates(list(stock.options))
            except Exception:
                results[ticker] = _emptyResult("Not enough option data.")
                continue

            options_chains   = {d: stock.option_chain(d) for d in exp_dates}
            price_history    = stock.history(period='3mo')
            underlying_price = price_history['Close'].iloc[-1]

            atm_iv   = {}
            straddle = None

            for i, (exp_date, chain) in enumerate(options_chains.items()):
                calls, puts = chain.calls, chain.puts
                if calls.empty or puts.empty:
                    continue

                call_idx = (calls['strike'] - underlying_price).abs().idxmin()
                put_idx  = (puts['strike']  - underlying_price).abs().idxmin()

                atm_iv[exp_date] = (
                    calls.loc[call_idx, 'impliedVolatility'] +
                    puts.loc[put_idx,  'impliedVolatility']
                ) / 2.0

                if i == 0:
                    call_bid, call_ask = calls.loc[call_idx, 'bid'], calls.loc[call_idx, 'ask']
                    put_bid,  put_ask  = puts.loc[put_idx,  'bid'], puts.loc[put_idx,  'ask']

                    call_mid = (call_bid + call_ask) / 2.0 if (call_bid is not None and call_ask is not None) else None
                    put_mid  = (put_bid  + put_ask)  / 2.0 if (put_bid  is not None and put_ask  is not None) else None

                    if call_mid and put_mid:
                        straddle = call_mid + put_mid
                    else:
                        call_last = float(calls.loc[call_idx, 'lastPrice'])
                        put_last  = float(puts.loc[put_idx,   'lastPrice'])
                        if not np.isnan(call_last) and not np.isnan(put_last) and call_last > 0 and put_last > 0:
                            straddle = call_last + put_last

            if not atm_iv:
                results[ticker] = _emptyResult("Could not determine ATM IV for any expiration dates.")
                continue

            today = datetime.today().date()
            dtes  = [(datetime.strptime(d, "%Y-%m-%d").date() - today).days for d in atm_iv]
            ivs   = list(atm_iv.values())

            curve         = buildIVCurve(dtes, ivs)
            ts_slope      = (curve(45) - curve(dtes[0])) / (45 - dtes[0])
            iv30_rv30     = curve(30) / yangZhang(price_history)
            avg_volume    = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
            expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else None

            vol_pass   = avg_volume >= 1_500_000
            iv_pass    = iv30_rv30  >= 1.25
            slope_pass = ts_slope   <= -0.00406

            if vol_pass and iv_pass and slope_pass:
                recommendation = 'Recommended'
            elif slope_pass and ((vol_pass and not iv_pass) or (iv_pass and not vol_pass)):
                recommendation = 'Consider'
            else:
                recommendation = 'Avoid'

            results[ticker] = {
                'recommendation': recommendation,
                'avg_volume':     vol_pass,
                'iv30_rv30':      iv_pass,
                'ts_slope_0_45':  slope_pass,
                'expected_move':  expected_move,
                'error':          None
            }

        except Exception as e:
            results[ticker] = _emptyResult(str(e))

    return results


def getEarningsTickers(date: str = None) -> list[str]:
    """
    Returns a list of tickers reporting earnings after market close (AMC).
    date: 'YYYY-MM-DD' format, defaults to today if not provided
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={date}"
    
    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Origin":          "https://www.nasdaq.com",
        "Referer":         "https://www.nasdaq.com/",
    }

    try:
        rows = requests.get(url, headers=headers).json().get("data", {}).get("rows", [])
        return [r["symbol"] for r in rows if r.get("time", "").strip() == "time-after-hours"]
    except Exception as e:
        print(f"Error fetching earnings calendar: {e}")
        return []


def scanEarnings(date: str = None) -> dict:
    tickers = getEarningsTickers(date)

    if not tickers:
        print("No AMC earnings found.")
        return {}

    print(f"Scanning {len(tickers)} tickers reporting AMC...\n")

    results = isTickerEligible(tickers)

    recommended = {}
    for ticker, result in results.items():
        em    = result.get('expected_move') or 'N/A'
        vol   = 'PASS' if result['avg_volume']   else 'FAIL'
        iv    = 'PASS' if result['iv30_rv30']     else 'FAIL'
        slope = 'PASS' if result['ts_slope_0_45'] else 'FAIL'
        error = f" | ERR: {result['error']}"      if result['error'] else ''

        print(f"{ticker:6} | {result['recommendation']:12} | EM: {em:>7} | Vol: {vol} | IV: {iv} | Slope: {slope}{error}")

        if result['recommendation'] == 'Recommended':
            recommended[ticker] = result

    print(f"\n{len(recommended)} Recommended setup(s) found.")
    return recommended


def getCalendarLegs(tickers) -> dict:
    """
    Accepts either a single ticker string, a list, or a dict of tickers.
    Returns a dict keyed by ticker with their calendar spread legs.
    """
    # normalize input to a list of tickers
    if isinstance(tickers, str):
        tickers = [tickers]
    elif isinstance(tickers, dict):
        tickers = list(tickers.keys())

    results = {}

    for ticker in tickers:
        try:
            ticker = ticker.strip().upper()
            stock  = yf.Ticker(ticker)

            if not stock.options:
                results[ticker] = {'error': f"No options found for '{ticker}'"}
                continue

            price_history    = stock.history(period='5d')
            underlying_price = price_history['Close'].iloc[-1]

            today    = datetime.today().date()
            tomorrow = today + timedelta(days=1)

            all_dates = sorted(
                datetime.strptime(d, "%Y-%m-%d").date()
                for d in stock.options
            )

            front_date = next((d for d in all_dates if d >= tomorrow), None)
            if front_date is None:
                results[ticker] = {'error': 'No valid front month expiration found'}
                continue

            target_back = front_date + timedelta(days=30)
            back_date   = min(
                (d for d in all_dates if d > front_date),
                key=lambda d: abs((d - target_back).days),
                default=None
            )
            if back_date is None:
                results[ticker] = {'error': 'No valid back month expiration found'}
                continue

            front_chain = stock.option_chain(front_date.strftime("%Y-%m-%d"))
            calls       = front_chain.calls

            if calls.empty:
                results[ticker] = {'error': 'No calls found for front month'}
                continue

            atm_idx    = (calls['strike'] - underlying_price).abs().idxmin()
            atm_strike = calls.loc[atm_idx, 'strike']

            def toOCC(ticker, date, strike, option_type='C'):
                expiry     = date.strftime("%y%m%d")
                strike_str = str(int(strike * 1000)).zfill(8)
                return f"{ticker}{expiry}{option_type}{strike_str}"

            front_symbol = toOCC(ticker, front_date, atm_strike)
            back_symbol  = toOCC(ticker, back_date,  atm_strike)

            results[ticker] = {
                'ticker':       ticker,
                'underlying':   round(underlying_price, 2),
                'atm_strike':   atm_strike,
                'front_date':   front_date.strftime("%Y-%m-%d"),
                'back_date':    back_date.strftime("%Y-%m-%d"),
                'front_dte':    (front_date - today).days,
                'back_dte':     (back_date  - today).days,
                'front_symbol': front_symbol,
                'back_symbol':  back_symbol,
                'error':        None
            }

            print(f"\n{ticker:6} | Underlying: ${round(underlying_price, 2):<8} | Strike: ${atm_strike:<8} | Front: {front_date} ({(front_date - today).days} DTE) | Back: {back_date} ({(back_date - today).days} DTE)")
            print(f"{'':6}   Sell: {front_symbol}")
            print(f"{'':6}   Buy:  {back_symbol}")

        except Exception as e:
            results[ticker] = {'error': str(e)}
            print(f"{ticker:6} | ERROR: {str(e)}")

    return results

getCalendarLegs(scanEarnings())
# next, choose strike price, name both options and place order.