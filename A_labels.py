import pytz
import asyncio
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from datetime import datetime
from simulator.alpha import Alpha
from simulator import quant_stats, indicators_cal
from simulator.general_utils import save_pickle, load_pickle

from data_service.data_master import DataMaster

def calculate_quantiles(data):
    return pd.qcut(data, 10, labels=False, duplicates='drop') + 1  # '+1' to make quantiles start from 1 instead of 0

class A_labels(Alpha):
    def __init__(
            self,
            trade_range=None,
            instruments=[],
            execrates=None,
            commrates=None,
            longswps=None,
            shortswps=None,
            dfs={},
            positional_inertia=0,
            specs={"quantile":((0.00,0.00),(0.85,1.00))},
    ):
        super().__init__(
            trade_range=trade_range,
            instruments=instruments,
            execrates=execrates,
            commrates=commrates,
            longswps=longswps,
            shortswps=shortswps,
            dfs=dfs,
            positional_inertia=positional_inertia
        )
        self.specs = specs
        self.sysname = 'A001'
        self.spy = DataMaster().get_equity_service().get_single_ohlcv(ticker='SPY', exchange='US', period_days=5000).set_index('datetime').tz_localize(pytz.UTC)
        self.smooth = False
        self.equal_weight = False
        self.use_quantile = False
        self.labels = {}  # Dictionary to store labels for each instrument
        
    def calculate_signals_for_instrument(self, inst) -> pd.Series:
        """
        Method to calculate buy/sell signals for a given instrument
        """
        # Retrieve the adjusted close prices for the instrument
        prices = self.dfs[inst]['adj_close']
        
        # Define the short and long windows for moving averages
        short_window = 20
        long_window = 50
        
        # Calculate the short and long moving averages
        sma_short = prices.rolling(window=short_window, min_periods=1).mean()
        sma_long = prices.rolling(window=long_window, min_periods=1).mean()
        
        # Generate signals: 1 for "buy" when short MA crosses above long MA, -1 for "sell" when the opposite happens
        signals = pd.Series(index=prices.index, data=np.nan)
        signals[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1  # Buy signal
        signals[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1  # Sell signal
        
        # Forward fill our buy/sell signals
        signals = signals.ffill().fillna(0)
        
        return signals
    
    def is_signal_active(self, inst):
        """ 
        Check if there's an active (non-expired) label for the instrument
        """
        if inst in self.labels and self.labels[inst]:
            _, _, vert_barrier = self.barriers[inst]
            if pd.Timestamp.today() < vert_barrier:
                return True
        return False

    async def compute_signals(self, index=None) -> None:
        """
        Compute signals for each instrument in the strategy
        Check if triple barrier is reached and apply labels
        """
        alphas = []
        for inst in self.instruments:
            inst_alpha = self.calculate_signals_for_instrument(inst)
            alphas.append(inst_alpha)
            
            # Check if a new signal to buy is generated and no active triple barrier for the instrument
            if inst_alpha[-1] == 1 and not self.is_signal_active(inst):
                prices = self.dfs[inst]['adj_close']
                daily_volatility = self.dfs[inst]['rets'].rolling(window=20).std()
                self.calculate_triple_barriers(inst, prices, daily_volatility)
                self.apply_labels(inst)

        alphadf = pd.concat(alphas, axis=1)
        alphadf.columns = self.instruments
        self.pad_ffill_dfs["alphadf"] = alphadf
    
    def calculate_quantiles(data):
        return pd.qcut(data, 10, labels=False, duplicates='drop') + 1  # '+1' to make quantiles start from 1 instead of 0
    
    def calculate_triple_barriers(self, instrument, prices, daily_volatility) -> None:
        """
        Calculate the upper, lower, and vertical barriers for the instrument
        :instrument: instrument to calculate barriers for
        :prices: adjusted close prices for the instrument
        :daily_volatility: daily volatility for the instrument
        :returns: None
        
        """
        # Calculate upper, lower, and vertical barriers based on the latest price and volatility
        # Assume vertical barrier duration (e.g., 10 days from the signal)
        t_final = pd.Timedelta(days=10)  # For example, 10 days
        last_price = prices.iloc[-1]
        vol = daily_volatility.iloc[-1]
        upper_barrier = last_price * (1 + 0.01 * vol)  # Example: 1% volatility for upper
        lower_barrier = last_price * (1 - 0.01 * vol)  # Example: 1% volatility for lower
        vert_barrier = prices.index[-1] + t_final

        # Store calculated barriers
        self.barriers[instrument] = (upper_barrier, lower_barrier, vert_barrier)
        # Apply label
        self.labels[instrument] = True  # Mark as active signal
        
    def update_barrier_status(self) -> None:
        """
        Check if the current price has reached any of the barriers or if the vertical barrier has been reached
        """
        for inst, (upper_barrier, lower_barrier, vert_barrier) in self.barriers.items():
            current_price = self.dfs[inst]['adj_close'].iloc[-1]
            if current_price >= upper_barrier or current_price <= lower_barrier or pd.Timestamp.today() >= vert_barrier:
                self.labels[inst] = False  # Deactivate signal
    
    def instantiate_eligibilities_and_strat_variables(self, eligiblesdf):
        """
        Create the eligibility DataFrame and other strategy variables
        """
        self.alphadf = self.pad_ffill_dfs["alphadf"]
        eligibles = []

        # Iterate through each instrument and date in alphadf to apply eligibility criteria
        for inst in self.instruments:
            # Initialize an empty series for eligibility with the same index as alphadf
            inst_eligible_series = pd.Series(False, index=self.alphadf.index)

            if self.is_signal_active(inst):
                # Make the entire series True if there's an active signal
                inst_eligible_series[:] = True
            else:
                # Apply default eligibility criteria when no active signal
                inst_eligible_series = (~pd.isna(self.alphadf[inst])) & \
                                        self.activedf[inst].astype("bool") & \
                                        (self.voldf[inst] > 0.00001).astype("bool") & \
                                        (self.baseclosedf[inst] > 0).astype("bool") & \
                                        (self.retdf[inst].abs() < 0.30).astype("bool")

            eligibles.append(inst_eligible_series)

        # Combine the eligibility series into a DataFrame and convert types
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.instruments
        self.eligiblesdf = self.eligiblesdf.astype("int8")
        
        return
    
    def compute_forecasts(self, portfolio_i, date, eligibles_row):
        """
        Transform signal to forecasts for market neutral strategy
        :portfolio_i: portfolio index
        :date: date
        :eligibles_row: eligible instruments np.array
        :equal_weight: wether equal wt or scale the forecast linearly to the factor
        :use_quantile: wether to use quantile or not
        """
        factor = self.alphadf.loc[date].values
        if self.equal_weight:
            factor_filtered = np.median(factor[eligibles_row == 1])
            factor_demean = np.nan_to_num(np.sign(factor - factor_filtered) * eligibles_row, nan=0, posinf=0, neginf=0)
            forecasts = (factor_demean)/np.sum(eligibles_row)
        else:
            factor_filtered = np.mean(factor[eligibles_row == 1])
            factor_demean = np.nan_to_num((factor - factor_filtered) * eligibles_row, nan=0, posinf=0, neginf=0)
            forecasts = factor_demean/np.sum(np.abs(factor_demean))

        return forecasts

    def factor_to_forecasts(self, factor, quantile, eligibles_row, eligible_quantile=None, equal_weight=False):
        """
        transform signal to forecasts for market neutral strategy
        :factor : signal np.array
        :quantile: cs factor quantile np array
        :eligibles_row: eligible instruments np.array
        :eligible_quantile: eligible quantiles to trade, list [1,5], default None
        :equal_weight: wether equal wt or scale the forecast linearly to the factor
        :return: forecast that will be used for position calculation
        """
        # update the eligibles
        if eligible_quantile is not None:
            for i in range(len(eligibles_row)):
                if not (quantile[i] in eligible_quantile):
                    eligibles_row[i] = 0
        factor_filtered = factor[eligibles_row == 1]
        if equal_weight:
            factor_demean = (np.sign(factor-np.median(factor_filtered))*eligibles_row)
            factor_demean = np.nan_to_num(factor_demean, nan=0, posinf=0, neginf=0)
            forecasts = (factor_demean)/np.sum(eligibles_row)
        else:
            factor_demean = (factor- np.mean(factor_filtered))*eligibles_row
            factor_demean = np.nan_to_num(factor_demean, nan=0, posinf=0, neginf=0)
            forecasts = factor_demean/np.sum(np.abs(factor_demean))

        return forecasts 
