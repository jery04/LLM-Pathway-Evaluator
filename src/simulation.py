"""
Monte Carlo learning path simulator.

Models student variability (learning speed, study hours, fatigue, dropout)
to estimate completion probability, time distributions, and bottlenecks.

Runs N simulations per path using:
- Log-normal durations
- Poisson weekly study capacity
- Weibull dropout hazard

Returns: completion_rate, mean/median/p10/p90 times, dropout_probability,
fatigue_risk, and course_bottlenecks dict.
"""

import numpy as np  # Statistical distributions and random sampling
from typing import List, Dict, Optional  # Type hints for function signatures
from dataclasses import dataclass    # Create structured data classes
from collections import defaultdict  # Auto-initialize nested dictionaries
import pandas as pd  # DataFrames for comparison tables and structured results


# ============================================================================
# PROBABILITY MODELS FOR LEARNING
# ============================================================================

@dataclass
class LearningParameters:
    """
    Parameters controlling the learning simulation model.
    Tune these to match different student profiles.
    """
    learning_rate_mu: float = 2.0      # Mean of log-normal learning rate
    learning_rate_sigma: float = 0.5   # Std deviation of log-normal learning rate
    forgetting_rate: float = 0.1       # Weekly knowledge decay rate
    weekly_capacity_lambda: float = 10  # Average study hours/week (Poisson)
    dropout_shape: float = 1.5         # Weibull shape - dropout curve steepness
    dropout_scale: float = 20          # Weibull scale - when dropout peaks
    fatigue_multiplier: float = 0.15   # Extra time penalty per difficult course
    recovery_rate: float = 0.3         # Weekly fatigue reduction rate

@dataclass
class SimulationResult:
    """Aggregated results from Monte Carlo simulation of a single path."""
    path_name: str
    path_index: int
    n_simulations: int
    completion_rate: float  # % of simulations that finished all courses
    mean_time_weeks: float
    median_time_weeks: float
    std_time_weeks: float
    p10_time_weeks: float  # 10th percentile - optimistic estimate
    p90_time_weeks: float  # 90th percentile - conservative estimate
    dropout_probability: float
    fatigue_risk: float  # Probability of chronic fatigue
    course_bottlenecks: Dict[str, float]  # Dropout rates per course
    completion_times: List[float]

class MonteCarloPathSimulator:
    """
    Monte Carlo simulator for learning paths.
    Models student heterogeneity through statistical distributions.
    """
    
    def __init__(self, course_index: Dict[str, Dict], random_seed: int = 42):
        """Initialize simulator with course data and fixed random seed for reproducibility."""
        self.course_index = course_index
        np.random.seed(random_seed)  # Ensure reproducible simulations
    
    def simulate_path(
        self,
        path: Dict,
        path_index: int,
        user_params: Optional[LearningParameters] = None,
        n_simulations: int = 1000
    ) -> SimulationResult:
        """Run Monte Carlo simulations for one learning path.
        Returns aggregated statistics as a SimulationResult.
        """
        
        params = user_params or LearningParameters()
        courses = path.get('path', [])  # Extract course sequence
        
        if not courses:
            return None
        
        completion_times = []
        course_failures = defaultdict(int)  # Track where students drop out
        
        for _ in range(n_simulations):
            result = self._run_single_simulation(courses, params)
            
            if result['completed']:
                completion_times.append(result['total_weeks'])
            else:
                course_failures[result['dropout_course']] += 1
        
        completion_rate = len(completion_times) / n_simulations
        
        # Calculate bottleneck probabilities per course
        course_bottlenecks = {
            course: failures / n_simulations 
            for course, failures in course_failures.items()
        }
        
        # Assess cumulative fatigue risk across all courses
        fatigue_risk = self._calculate_fatigue_risk(courses, params)
        
        if completion_times:
            times = np.array(completion_times)
            return SimulationResult(
                path_name=path.get('target_course', 'Unknown'),
                path_index=path_index,
                n_simulations=n_simulations,
                completion_rate=completion_rate,
                mean_time_weeks=np.mean(times),
                median_time_weeks=np.median(times),
                std_time_weeks=np.std(times),
                p10_time_weeks=np.percentile(times, 10),
                p90_time_weeks=np.percentile(times, 90),
                dropout_probability=1 - completion_rate,
                fatigue_risk=fatigue_risk,
                course_bottlenecks=course_bottlenecks,
                completion_times=times.tolist()
            )
        return None
    
    def _run_single_simulation(self, courses: List[str], params: LearningParameters) -> Dict:
        """Simulate a single student through the course sequence.
        Returns a dict with completion flag, dropout point, and total weeks.
        """
        
        total_weeks = 0
        fatigue_level = 0.0  # 0 = rested, 1 = exhausted
        
        for course_idx, course_name in enumerate(courses):
            course = self.course_index.get(course_name, {})
            
            # Sample course duration accounting for learning rate variability
            base_weeks = self._sample_course_duration(course, params)
            
            # Apply fatigue penalty (tired students take longer)
            fatigue_penalty = 1 + fatigue_level * params.fatigue_multiplier
            adjusted_weeks = base_weeks * fatigue_penalty
            
            # Sample actual weekly study capacity (varies week to week)
            weekly_capacity = max(1, np.random.poisson(params.weekly_capacity_lambda))
            course_weeks = adjusted_weeks * (40 / weekly_capacity)
            
            # Check if student drops out before finishing this course
            dropout_prob = self._calculate_dropout_probability(
                course_idx, len(courses), fatigue_level, params
            )
            
            if np.random.random() < dropout_prob:
                return {
                    'completed': False,
                    'dropout_course': course_name,
                    'total_weeks': total_weeks
                }
            
            # Update fatigue based on course difficulty
            course_difficulty = course.get('difficulty', 5) / 10  # Scale 0-1
            fatigue_level = min(1.0, fatigue_level + course_difficulty * params.fatigue_multiplier)
            
            # Take recovery break between courses
            recovery_weeks = self._sample_recovery_time(fatigue_level, params)
            total_weeks += course_weeks + recovery_weeks
            fatigue_level = max(0, fatigue_level - params.recovery_rate * recovery_weeks)
        
        return {'completed': True, 'total_weeks': total_weeks}
    
    def _sample_course_duration(self, course: Dict, params: LearningParameters) -> float:
        """Sample a course duration in weeks using a log-normal model.
        Accounts for base duration and learner variability.
        """
        base_months = course.get('duration_months', 1)
        base_hours = base_months * 40  # 40 hours = 1 week of full-time study
        
        # Log-normal models multiplicative learning rate effects
        Z = np.random.normal(0, 1)
        log_duration = np.log(base_hours) + params.learning_rate_mu + params.learning_rate_sigma * Z
        sampled_weeks = np.exp(log_duration) / 40  # Convert back to weeks
        
        return max(0.5, sampled_weeks)  # Minimum 0.5 weeks
    
    def _calculate_dropout_probability(self, course_idx: int, total_courses: int, 
                                        fatigue_level: float, params: LearningParameters) -> float:
        """Estimate dropout probability for a course using a Weibull hazard.
        Incorporates course position and current fatigue level.
        """
        position_factor = np.exp(-course_idx / 3)  # Early courses safer
        fatigue_factor = 1 + fatigue_level * 2  # Fatigue doubles risk
        
        t = course_idx + 1  # Course number (1-indexed)
        # Weibull hazard: increases with course index (cumulative effort)
        hazard = (params.dropout_shape / params.dropout_scale) * \
                 (t / params.dropout_scale) ** (params.dropout_shape - 1)
        
        return min(0.3, hazard * position_factor * fatigue_factor)  # Cap at 30%
    
    def _sample_recovery_time(self, fatigue_level: float, params: LearningParameters) -> float:
        """Sample recovery weeks needed based on current fatigue level.
        Returns a small number of rest weeks (capped).
        """
        if fatigue_level < 0.3:
            return 0  # No rest needed for low fatigue
        needed_weeks = fatigue_level * 2 * np.random.exponential(1)  # Exponential rest duration
        return min(4, needed_weeks)  # Maximum 4 weeks recovery
    
    def _calculate_fatigue_risk(self, courses: List[str], params: LearningParameters) -> float:
        """Compute a fatigue risk score from average course difficulty.
        Returns a probability between 0 and 1.
        """
        total_difficulty = 0
        for course_name in courses:
            course = self.course_index.get(course_name, {})
            total_difficulty += course.get('difficulty', 5)
        
        avg_difficulty = total_difficulty / len(courses) if courses else 5
        # Scale difficulty (3=low risk, 10=high risk) to probability
        return min(1.0, (avg_difficulty - 3) / 7)


# ============================================================================
# SIMULATION ORCHESTRATION FUNCTION
# ============================================================================

def simulate_paths(course_index: Dict, paths: List[Dict], n_simulations: int = 500):
    """Run Monte Carlo simulations for multiple learning paths.
    Returns a list of SimulationResult objects for each path.
    """
    
    if not paths:
        print("❌ No paths to simulate")
        return None
    
    print(f"\n🎲 Starting Monte Carlo simulation")
    print(f"📊 {n_simulations} simulations per path")
    print(f"📚 {len(paths)} paths to analyze")
    print("-" * 50)
    
    simulator = MonteCarloPathSimulator(course_index)
    results = []
    
    for i, path in enumerate(paths, 1):
        print(f"\n🔄 Simulating Path #{i}: {path.get('target_course', 'Unknown')[:50]}...")
        
        # Customizable parameters for different student profiles
        params = LearningParameters(
            weekly_capacity_lambda=15,  # Average 15 hours/week study
            fatigue_multiplier=0.15,    # Normal fatigue sensitivity
            dropout_scale=20            # Medium commitment level
        )
        
        result = simulator.simulate_path(path, i, params, n_simulations)
        
        if result:
            results.append(result)
            print(f"   ✅ Success rate: {result.completion_rate:.1%} | "
                  f"Time: {result.mean_time_weeks:.0f} weeks")
        else:
            print(f"   ⚠️ Could not simulate")
    
    return results


# ============================================================================
# UI INTEGRATION & VISUALIZATION LAYER
# ============================================================================

def get_simulation_ui_data(
    course_index: Dict,
    paths: List[Dict],
    user_params: LearningParameters,
    n_simulations: int,
) -> Dict:
    """Run simulations for UI and return aggregated data dict.
    Contains results, time distributions, bottlenecks and summary stats.
    """
    simulator = MonteCarloPathSimulator(course_index)
    results = []
    time_distributions = {}

    for i, path in enumerate(paths, 1):
        res = simulator.simulate_path(path, i, user_params, n_simulations)
        if res:
            results.append(res)
            time_distributions[i - 1] = res.completion_times

    # Aggregate course-level bottlenecks across all paths
    total_failures = defaultdict(float)
    total_simulations = 0
    for res in results:
        total_simulations += res.n_simulations
        for course, rate in (res.course_bottlenecks or {}).items():
            total_failures[course] += rate * res.n_simulations

    course_bottlenecks = {
        course: (failures / total_simulations) if total_simulations else 0.0
        for course, failures in total_failures.items()
    }

    # Choose best path by highest completion_rate (tie-breaker: lower mean time)
    best_index = 0
    if results:
        best = max(results, key=lambda r: (r.completion_rate, -r.mean_time_weeks))
        best_index = best.path_index - 1

    # Summary stats
    summary_stats = {
        'avg_completion_rate': float(np.mean([r.completion_rate for r in results])) if results else 0.0,
        'avg_time_weeks': float(np.mean([r.mean_time_weeks for r in results])) if results else 0.0,
        'avg_dropout_prob': float(np.mean([r.dropout_probability for r in results])) if results else 0.0,
        'avg_fatigue_risk': float(np.mean([r.fatigue_risk for r in results])) if results else 0.0,
    }

    return {
        'results': results,
        'time_distributions': time_distributions,
        'course_bottlenecks': course_bottlenecks,
        'best_path_index': best_index,
        'summary_stats': summary_stats,
    }

def create_comparison_table(results: List[SimulationResult]) -> 'pd.DataFrame':
    """Build a pandas DataFrame summarizing simulation results.
    Formats numeric columns for display.
    """

    rows = []
    for r in results:
        rows.append({
            'path_index': r.path_index,
            'path_name': r.path_name,
            'completion_rate': r.completion_rate,
            'mean_time_weeks': r.mean_time_weeks,
            'median_time_weeks': r.median_time_weeks,
            'std_time_weeks': r.std_time_weeks,
            'p10_time_weeks': r.p10_time_weeks,
            'p90_time_weeks': r.p90_time_weeks,
            'dropout_probability': r.dropout_probability,
            'fatigue_risk': r.fatigue_risk,
            'n_simulations': r.n_simulations,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['completion_rate', 'mean_time_weeks'], ascending=[False, True])
        # Use round on columns directly to avoid potential applymap issues across pandas versions
        for c in ['completion_rate', 'dropout_probability', 'fatigue_risk']:
            if c in df.columns:
                df[c] = df[c].astype(float).round(3)

        for c in ['mean_time_weeks', 'median_time_weeks', 'std_time_weeks', 'p10_time_weeks', 'p90_time_weeks']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').round(1)

    return df

def plot_time_distribution(times: List[float], path_name: str) -> 'plt.Figure':
    """Return a matplotlib figure with a time-to-completion histogram.
    Shows KDE and density for provided times.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 4))
    if not times:
        ax.text(0.5, 0.5, 'No simulation data', ha='center', va='center')
        return fig

    sns.histplot(times, kde=True, ax=ax, stat='density', color='#4a78b8', edgecolor='k')
    ax.set_title(f'Time distribution — {path_name}')
    ax.set_xlabel('Weeks to completion')
    ax.set_ylabel('Density')
    plt.tight_layout()
    return fig

def plot_time_boxplot(times: List[float], path_name: str) -> 'plt.Figure':
    """Return a matplotlib boxplot figure for completion times.
    Useful to display spread and outliers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 3))
    if not times:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    sns.boxplot(x=times, ax=ax, color='#8ab8ff')
    ax.set_title(f'Completion time spread — {path_name}')
    ax.set_xlabel('Weeks')
    plt.tight_layout()
    return fig

def plot_time_cdf(times: List[float], path_name: str) -> 'plt.Figure':
    """Return a matplotlib CDF plot for completion times.
    Displays cumulative completion probability over weeks.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 3))
    if not times:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    data = np.sort(np.array(times))
    y = np.arange(1, len(data)+1) / len(data)
    ax.step(data, y, where='post', color='#4a78b8')
    ax.set_title(f'Cumulative completion (CDF) — {path_name}')
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Cumulative probability')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig
