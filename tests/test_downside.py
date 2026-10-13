import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from downside import label_regimes, signal_state


class RegimeTests(unittest.TestCase):
    def test_thresholds_and_crash_flag(self):
        prices = pd.Series([100, 100, 95, 89, 82, 79, 85], index=pd.period_range('2000-01', periods=7, freq='M'))
        reg = label_regimes(prices)
        self.assertEqual(list(reg.regime), ['normal', 'normal', 'normal', 'correction', 'correction', 'bear', 'correction'])
        crashy = label_regimes(pd.Series([100, 85], index=pd.period_range('2000-01', periods=2, freq='M')))
        self.assertTrue(crashy.crash_month.iloc[1])
        self.assertEqual(signal_state(crashy).iloc[1], 'post_crash')


if __name__ == '__main__':
    unittest.main()
