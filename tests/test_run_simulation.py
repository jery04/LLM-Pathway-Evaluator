"""Unit tests for simulation module.

This file provides concise tests for the public behavior of the
`create_comparison_table` helper using `SimulationResult` instances.
"""
import sys  # System path manipulation for module imports
import os   # File system path operations
import unittest  # Unit testing framework

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation import SimulationResult  # Dataclass for aggregated simulation stats
import simulation as sim  # Monte Carlo simulation module alias


class TestRunSimulation(unittest.TestCase):
    """Basic tests for simulation.create_comparison_table"""

    def make_result(self, name, idx, months, completion_rate=0.8):
        """Create a SimulationResult with sensible defaults for tests."""
        weeks = months * 4.0
        return SimulationResult(
            path_name=name,
            path_index=idx,
            n_simulations=100,
            completion_rate=completion_rate,
            mean_time_weeks=weeks,
            median_time_weeks=weeks,
            std_time_weeks=2.0,
            p10_time_weeks=weeks * 0.75,
            p90_time_weeks=weeks * 1.25,
            dropout_probability=1.0 - completion_rate,
            fatigue_risk=0.1 + idx * 0.01,
            course_bottlenecks={"Course A": 0.1},
            completion_times=[weeks + i for i in range(10)],
        )

    def test_create_comparison_table_returns_dataframe_like(self):
        """Return a dataframe-like object with expected columns and rows."""
        r1 = self.make_result('Path A', 1, 1.0, completion_rate=0.9)
        r2 = self.make_result('Path B', 2, 2.0, completion_rate=0.7)

        df = sim.create_comparison_table([r1, r2])

        # Sanity checks: tiene filas igual al número de resultados y columnas esperadas
        self.assertTrue(hasattr(df, 'shape'))
        self.assertEqual(df.shape[0], 2)

        # Columnas mínimas que esperamos: path_name, completion_rate, mean_time_weeks
        for col in ('path_name', 'completion_rate', 'mean_time_weeks'):
            self.assertIn(col, df.columns)

    def test_create_comparison_table_handles_empty_list(self):
        """Handle an empty input list without raising exceptions."""
        df = sim.create_comparison_table([])
        # Esperamos un DataFrame vacío (0 filas)
        self.assertTrue(hasattr(df, 'shape'))
        self.assertEqual(df.shape[0], 0)

    def test_column_types(self):
        """Ensure key columns have expected numeric types where applicable."""
        r = self.make_result('Single', 1, 1.0, completion_rate=0.85)
        df = sim.create_comparison_table([r])
        self.assertTrue(df['completion_rate'].dtype.kind in ('f', 'i'))
        self.assertTrue(df['mean_time_weeks'].dtype.kind in ('f', 'i'))

    def test_completion_rate_bounds(self):
        """Completion rates should be within [0, 1]."""
        r = self.make_result('Bounds', 1, 1.0, completion_rate=0.0)
        r2 = self.make_result('Bounds2', 2, 1.0, completion_rate=1.0)
        df = sim.create_comparison_table([r, r2])
        self.assertGreaterEqual(df['completion_rate'].min(), 0.0)
        self.assertLessEqual(df['completion_rate'].max(), 1.0)

    def test_mean_time_values(self):
        """Mean time should match the provided SimulationResult value."""
        r = self.make_result('TimeCheck', 1, 3.0, completion_rate=0.5)
        df = sim.create_comparison_table([r])
        self.assertAlmostEqual(df.loc[0, 'mean_time_weeks'], r.mean_time_weeks)

    def test_handles_missing_fields(self):
        """Rows with missing optional fields should still be represented."""
        # Construct a minimal SimulationResult-like object by attribute assignment
        class Minimal:
            pass

        m = Minimal()
        m.path_name = 'Minimal'
        m.path_index = 1
        m.n_simulations = 10
        m.completion_rate = 0.5
        m.mean_time_weeks = 4.0
        m.median_time_weeks = 4.0
        m.std_time_weeks = None
        m.p10_time_weeks = None
        m.p90_time_weeks = None
        m.dropout_probability = 0.5
        m.fatigue_risk = None
        m.course_bottlenecks = {}
        m.completion_times = []

        df = sim.create_comparison_table([m])
        self.assertEqual(df.shape[0], 1)

    def test_handles_none_result(self):
        """If non-result entries are present, function should raise an AttributeError."""
        r = self.make_result('Valid', 1, 1.0)
        with self.assertRaises(AttributeError):
            sim.create_comparison_table([None, r, None])

    def test_columns_non_null(self):
        """Important identifier columns should not be null."""
        r = self.make_result('NonNull', 1, 1.0)
        df = sim.create_comparison_table([r])
        self.assertFalse(df['path_name'].isnull().any())
        self.assertFalse(df['path_index'].isnull().any())

    def test_consistent_dropout_probability(self):
        """Dropout probability should equal 1 - completion_rate when provided so."""
        r = self.make_result('Drop', 1, 1.0, completion_rate=0.6)
        df = sim.create_comparison_table([r])
        self.assertAlmostEqual(df.loc[0, 'dropout_probability'], 1.0 - df.loc[0, 'completion_rate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
