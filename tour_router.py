#!/usr/bin/env python3
"""
Tour Route Optimizer
====================
Builds optimized tour itineraries that maximize revenue and capacity utilization
starting from a specific city and expanding geographically.

Features:
- Geographic routing optimization
- Market saturation avoidance
- Distance-based travel optimization
- Multi-city tour planning
- Revenue and utilization scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TourRouter:
    def __init__(self, data_path='us_only_with_states.csv'):
        """Initialize the tour router with historical data"""
        print("="*80)
        print("TOUR ROUTE OPTIMIZER")
        print("="*80)
        
        print("\nLoading data...")
        self.df = pd.read_csv(data_path)
        self.df['Event Date'] = pd.to_datetime(self.df['Event Date'], errors='coerce')
        print(f"✓ Loaded {len(self.df):,} events")
        
        # Build venue database
        self._build_venue_database()
        
        # Build city coordinates (approximate state capitals)
        self._initialize_city_coordinates()
        
        print("\n✓ Tour router initialized")
    
    def _build_venue_database(self):
        """Build comprehensive venue database with performance metrics"""
        print("\nBuilding venue database...")
        
        self.venues = self.df.groupby(['Venue', 'City', 'State']).agg({
            'Revenue (USD)': ['mean', 'sum', 'count'],
            'Capacity Utilization (%)': 'mean',
            'Avg Ticket Price': 'mean',
            'Capacity': 'first',
            'Tickets Sold': 'sum',
            'Genre': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        self.venues.columns = [
            'Venue', 'City', 'State',
            'Avg_Revenue', 'Total_Revenue', 'Event_Count',
            'Avg_Capacity_Util', 'Avg_Ticket_Price',
            'Capacity', 'Total_Tickets', 'Top_Genre'
        ]
        
        # Calculate performance score
        max_revenue = self.venues['Avg_Revenue'].max()
        self.venues['Revenue_Score'] = self.venues['Avg_Revenue'] / max_revenue
        self.venues['Utilization_Score'] = self.venues['Avg_Capacity_Util'] / 100
        self.venues['Performance_Score'] = (
            0.6 * self.venues['Revenue_Score'] + 
            0.4 * self.venues['Utilization_Score']
        )
        
        print(f"✓ Database built: {len(self.venues)} venues")
    
    def _initialize_city_coordinates(self):
        """Initialize approximate coordinates for cities (state-level)"""
        # Approximate coordinates of state centers for distance calculation
        self.state_coords = {
            'Alabama': (32.8, -86.8), 'Alaska': (64.0, -152.0), 'Arizona': (34.3, -111.7),
            'Arkansas': (34.9, -92.4), 'California': (37.0, -120.0), 'Colorado': (39.0, -105.5),
            'Connecticut': (41.6, -72.7), 'Delaware': (39.0, -75.5), 'Florida': (28.5, -82.0),
            'Georgia': (32.7, -83.5), 'Hawaii': (20.0, -157.0), 'Idaho': (44.5, -114.0),
            'Illinois': (40.0, -89.0), 'Indiana': (40.0, -86.3), 'Iowa': (42.0, -93.5),
            'Kansas': (38.5, -98.5), 'Kentucky': (37.5, -85.0), 'Louisiana': (31.0, -92.0),
            'Maine': (45.5, -69.0), 'Maryland': (39.0, -76.7), 'Massachusetts': (42.3, -71.8),
            'Michigan': (44.3, -85.5), 'Minnesota': (46.0, -94.5), 'Mississippi': (32.7, -89.5),
            'Missouri': (38.5, -92.5), 'Montana': (47.0, -110.0), 'Nebraska': (41.5, -99.8),
            'Nevada': (39.5, -117.0), 'New Hampshire': (43.7, -71.5), 'New Jersey': (40.2, -74.7),
            'New Mexico': (34.5, -106.0), 'New York': (43.0, -75.0), 'North Carolina': (35.5, -80.0),
            'North Dakota': (47.5, -100.5), 'Ohio': (40.5, -82.5), 'Oklahoma': (35.5, -97.5),
            'Oregon': (44.0, -120.5), 'Pennsylvania': (41.0, -77.5), 'Rhode Island': (41.7, -71.5),
            'South Carolina': (34.0, -81.0), 'South Dakota': (44.5, -100.5), 'Tennessee': (35.8, -86.5),
            'Texas': (31.5, -99.0), 'Utah': (39.5, -111.5), 'Vermont': (44.0, -72.7),
            'Virginia': (37.5, -79.0), 'Washington': (47.5, -120.5), 'West Virginia': (38.5, -80.5),
            'Wisconsin': (44.5, -90.0), 'Wyoming': (43.0, -107.5), 'District of Columbia': (38.9, -77.0)
        }
    
    def _calculate_distance(self, state1, state2):
        """Calculate approximate distance between states (in arbitrary units)"""
        if state1 not in self.state_coords or state2 not in self.state_coords:
            return 999  # Large penalty for unknown states
        
        lat1, lon1 = self.state_coords[state1]
        lat2, lon2 = self.state_coords[state2]
        
        # Simple Euclidean distance (not actual miles, but proportional)
        distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        return distance
    
    def build_tour_route(self, genre, starting_city, starting_state, 
                        num_stops=10, min_capacity=None, max_capacity=None,
                        optimize_for='balanced', max_distance_penalty=True):
        """
        Build an optimized tour route starting from a specific city
        
        Parameters:
        -----------
        genre : str
            Genre for the tour (filters venues by genre affinity)
        starting_city : str
            Starting city name
        starting_state : str
            Starting state (2-letter code or full name)
        num_stops : int, default=10
            Number of tour stops to recommend
        min_capacity : int, optional
            Minimum venue capacity
        max_capacity : int, optional
            Maximum venue capacity
        optimize_for : str, default='balanced'
            'revenue', 'utilization', or 'balanced'
        max_distance_penalty : bool, default=True
            Apply penalty for long distances between stops
        
        Returns:
        --------
        pd.DataFrame : Optimized tour route with venues and metrics
        """
        
        print("\n" + "="*80)
        print(f"BUILDING TOUR ROUTE: {genre.upper()} TOUR")
        print("="*80)
        
        print(f"\nStarting point: {starting_city}, {starting_state}")
        print(f"Optimization: {optimize_for}")
        print(f"Target stops: {num_stops}")
        
        # Filter venues by genre affinity
        genre_venues = self._filter_by_genre(genre)
        
        # Apply capacity filters
        if min_capacity:
            genre_venues = genre_venues[genre_venues['Capacity'] >= min_capacity]
            print(f"Min capacity filter: {min_capacity:,}")
        
        if max_capacity:
            genre_venues = genre_venues[genre_venues['Capacity'] <= max_capacity]
            print(f"Max capacity filter: {max_capacity:,}")
        
        print(f"\n✓ Filtered to {len(genre_venues)} venues")
        
        # Calculate genre affinity scores
        genre_venues = self._calculate_genre_affinity(genre_venues, genre)
        
        # Set optimization weights
        if optimize_for == 'revenue':
            weights = {'revenue': 0.7, 'utilization': 0.1, 'affinity': 0.1, 'distance': 0.1}
        elif optimize_for == 'utilization':
            weights = {'revenue': 0.1, 'utilization': 0.7, 'affinity': 0.1, 'distance': 0.1}
        else:  # balanced
            weights = {'revenue': 0.35, 'utilization': 0.35, 'affinity': 0.15, 'distance': 0.15}
        
        # Build tour route iteratively
        tour_stops = []
        visited_states = set()
        current_state = starting_state
        
        # Find starting venue
        starting_venues = genre_venues[
            (genre_venues['City'] == starting_city) & 
            (genre_venues['State'] == starting_state)
        ]
        
        if len(starting_venues) == 0:
            print(f"\n⚠ No venues found in {starting_city}, {starting_state}")
            print("Finding nearest alternative starting point...")
            # Find venues in starting state
            starting_venues = genre_venues[genre_venues['State'] == starting_state]
            if len(starting_venues) == 0:
                print(f"⚠ No venues in {starting_state}, using best overall venue")
                starting_venues = genre_venues.nlargest(1, 'Performance_Score')
        
        # Select best starting venue
        starting_venue = starting_venues.nlargest(1, 'Performance_Score').iloc[0]
        tour_stops.append(starting_venue)
        visited_states.add(starting_venue['State'])
        current_state = starting_venue['State']
        
        print(f"\n✓ Starting venue: {starting_venue['Venue']} ({starting_venue['City']}, {starting_venue['State']})")
        
        # Build rest of tour
        remaining_venues = genre_venues.copy()
        
        for stop_num in range(1, num_stops):
            # Remove already selected venues
            remaining_venues = remaining_venues[
                ~remaining_venues['Venue'].isin([v['Venue'] for v in tour_stops])
            ]
            
            if len(remaining_venues) == 0:
                print(f"\n⚠ No more venues available (found {len(tour_stops)} stops)")
                break
            
            # Calculate scores for remaining venues
            remaining_venues = remaining_venues.copy()
            remaining_venues['Distance_Score'] = remaining_venues['State'].apply(
                lambda s: 1 - min(self._calculate_distance(current_state, s) / 50, 1)
            )
            
            # Bonus for new states (market diversity)
            remaining_venues['State_Diversity_Bonus'] = remaining_venues['State'].apply(
                lambda s: 0.1 if s not in visited_states else 0
            )
            
            # Calculate final score for next stop
            remaining_venues['Tour_Score'] = (
                weights['revenue'] * remaining_venues['Revenue_Score'] +
                weights['utilization'] * remaining_venues['Utilization_Score'] +
                weights['affinity'] * remaining_venues['Genre_Affinity_Score'] +
                weights['distance'] * remaining_venues['Distance_Score'] +
                remaining_venues['State_Diversity_Bonus']
            )
            
            # Select next venue
            next_venue = remaining_venues.nlargest(1, 'Tour_Score').iloc[0]
            tour_stops.append(next_venue)
            visited_states.add(next_venue['State'])
            current_state = next_venue['State']
            
            print(f"  Stop {stop_num + 1}: {next_venue['Venue']} ({next_venue['City']}, {next_venue['State']}) - Score: {next_venue['Tour_Score']:.3f}")
        
        # Convert to DataFrame
        tour_df = pd.DataFrame(tour_stops)
        tour_df.index = range(1, len(tour_df) + 1)
        tour_df.index.name = 'Stop'
        
        # Calculate tour statistics
        self._print_tour_summary(tour_df, genre)
        
        return tour_df
    
    def _filter_by_genre(self, genre):
        """Filter venues by genre affinity"""
        # Get venues that have hosted this genre
        genre_events = self.df[
            self.df['Genre'].str.lower() == genre.lower()
        ] if 'Genre' in self.df.columns else self.df
        
        if len(genre_events) == 0:
            print(f"⚠ No historical events found for genre '{genre}', using all venues")
            return self.venues.copy()
        
        # Get venues from genre events
        genre_venue_names = genre_events['Venue'].unique()
        genre_venues = self.venues[self.venues['Venue'].isin(genre_venue_names)].copy()
        
        if len(genre_venues) == 0:
            print(f"⚠ No matching venues found for genre '{genre}', using all venues")
            return self.venues.copy()
        
        return genre_venues
    
    def _calculate_genre_affinity(self, venues, genre):
        """Calculate how well each venue performs with the specified genre"""
        venues = venues.copy()
        
        # Get genre-specific performance
        genre_events = self.df[self.df['Genre'].str.lower() == genre.lower()] if 'Genre' in self.df.columns else pd.DataFrame()
        
        if len(genre_events) == 0:
            venues['Genre_Affinity_Score'] = 0.5
            return venues
        
        # Calculate performance by venue for this genre
        genre_perf = genre_events.groupby('Venue').agg({
            'Revenue (USD)': 'mean',
            'Capacity Utilization (%)': 'mean'
        }).reset_index()
        
        genre_perf.columns = ['Venue', 'Genre_Revenue', 'Genre_Utilization']
        
        # Normalize scores
        max_genre_rev = genre_perf['Genre_Revenue'].max()
        if max_genre_rev > 0:
            genre_perf['Genre_Rev_Score'] = genre_perf['Genre_Revenue'] / max_genre_rev
        else:
            genre_perf['Genre_Rev_Score'] = 0
        
        genre_perf['Genre_Util_Score'] = genre_perf['Genre_Utilization'] / 100
        
        genre_perf['Genre_Affinity_Score'] = (
            0.5 * genre_perf['Genre_Rev_Score'] +
            0.5 * genre_perf['Genre_Util_Score']
        )
        
        # Merge with venues
        venues = venues.merge(
            genre_perf[['Venue', 'Genre_Affinity_Score']],
            on='Venue',
            how='left'
        )
        venues['Genre_Affinity_Score'] = venues['Genre_Affinity_Score'].fillna(0.3)
        
        return venues
    
    def _print_tour_summary(self, tour_df, genre):
        """Print summary statistics for the tour"""
        print("\n" + "="*80)
        print("TOUR SUMMARY")
        print("="*80)
        
        total_revenue = tour_df['Avg_Revenue'].sum()
        avg_revenue_per_stop = tour_df['Avg_Revenue'].mean()
        avg_capacity_util = tour_df['Avg_Capacity_Util'].mean()
        total_capacity = tour_df['Capacity'].sum()
        avg_ticket_price = tour_df['Avg_Ticket_Price'].mean()
        states_covered = tour_df['State'].nunique()
        
        print(f"\nGenre: {genre}")
        print(f"Total Stops: {len(tour_df)}")
        print(f"States Covered: {states_covered}")
        print(f"\nRevenue Projections:")
        print(f"  Estimated Total Revenue:    ${total_revenue:,.0f}")
        print(f"  Avg Revenue per Stop:       ${avg_revenue_per_stop:,.0f}")
        print(f"  Total Capacity:             {total_capacity:,} seats")
        print(f"\nPerformance Metrics:")
        print(f"  Avg Capacity Utilization:   {avg_capacity_util:.1f}%")
        print(f"  Avg Ticket Price:           ${avg_ticket_price:.2f}")
        print(f"  Avg Performance Score:      {tour_df['Performance_Score'].mean():.3f}")
        
        print(f"\nStates on Tour:")
        state_counts = tour_df['State'].value_counts()
        for state, count in state_counts.items():
            print(f"  {state}: {count} stop(s)")
    
    def export_tour(self, tour_df, filename='tour_route.csv'):
        """Export tour route to CSV"""
        output_cols = [
            'Venue', 'City', 'State', 'Capacity',
            'Avg_Revenue', 'Avg_Capacity_Util', 'Avg_Ticket_Price',
            'Performance_Score', 'Event_Count'
        ]
        
        tour_export = tour_df[output_cols].copy()
        tour_export.to_csv(f'/mnt/user-data/outputs/{filename}', index=True)
        print(f"\n✓ Tour exported to {filename}")
        
        return tour_export
    
    def visualize_tour_map(self, tour_df):
        """Create a simple text-based visualization of the tour route"""
        print("\n" + "="*80)
        print("TOUR ROUTE MAP")
        print("="*80)
        
        print("\nRoute Sequence:")
        for idx, row in tour_df.iterrows():
            arrow = "    ↓" if idx < len(tour_df) else ""
            print(f"  {idx}. {row['Venue']}")
            print(f"     {row['City']}, {row['State']}")
            print(f"     Capacity: {row['Capacity']:,} | Util: {row['Avg_Capacity_Util']:.1f}% | Rev: ${row['Avg_Revenue']:,.0f}")
            if arrow:
                print(arrow)


def main():
    """Example usage: Build tour routes"""
    
    # Initialize router
    router = TourRouter('us_only_with_states.csv')
    
    # Example 1: Country music tour starting from Nashville
    print("\n" + "="*80)
    print("EXAMPLE 1: Country Music Tour from Nashville")
    print("="*80)
    
    country_tour = router.build_tour_route(
        genre='country',
        starting_city='Nashville',
        starting_state='Tennessee',
        num_stops=12,
        min_capacity=5000,
        optimize_for='balanced'
    )
    
    # Visualize route
    router.visualize_tour_map(country_tour)
    
    # Export tour
    router.export_tour(country_tour, 'country_tour_nashville.csv')
    
    # Example 2: R&B tour starting from Los Angeles
    print("\n" + "="*80)
    print("EXAMPLE 2: R&B Tour from Los Angeles")
    print("="*80)
    
    rb_tour = router.build_tour_route(
        genre='r&b',
        starting_city='Los Angeles',
        starting_state='California',
        num_stops=10,
        min_capacity=8000,
        max_capacity=25000,
        optimize_for='revenue'
    )
    
    router.visualize_tour_map(rb_tour)
    router.export_tour(rb_tour, 'rb_tour_la.csv')
    
    # Example 3: K-pop tour starting from New York
    print("\n" + "="*80)
    print("EXAMPLE 3: K-pop Tour from New York")
    print("="*80)
    
    kpop_tour = router.build_tour_route(
        genre='k-pop',
        starting_city='New York',
        starting_state='New York',
        num_stops=15,
        optimize_for='utilization'
    )
    
    router.visualize_tour_map(kpop_tour)
    router.export_tour(kpop_tour, 'kpop_tour_nyc.csv')
    
    print("\n" + "="*80)
    print("✅ TOUR ROUTING COMPLETE!")
    print("="*80)
    
    print("\nFiles created:")
    print("  • country_tour_nashville.csv")
    print("  • rb_tour_la.csv")
    print("  • kpop_tour_nyc.csv")


if __name__ == "__main__":
    main()
