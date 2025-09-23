"""
KPI Compliance Testing Suite
Validates 100% recall accuracy + <200ms performance requirements
"""

import time
import pandas as pd
from typing import List, Dict, Tuple
import statistics
from property_search_engine import PropertySearchEngine


class KPITestSuite:
    """Comprehensive testing to validate POC success metrics"""

    def __init__(self, csv_path: str):
        print("Initializing KPI Test Suite...")
        self.engine = PropertySearchEngine(csv_path)
        self.test_results = {}

    def test_100_percent_recall(self) -> bool:
        """
        KPI Test 1: 100% recall for obvious duplicates
        Tests exact matches (same address/specs) are always found
        """
        print("\n Testing 100% Recall Requirement...")

        # Create test cases from existing data (guaranteed duplicates)
        if self.engine.properties_df is None:
            raise RuntimeError("Properties dataframe is not loaded")
        sample_properties = self.engine.properties_df.sample(50, random_state=42)

        recall_tests = []
        for _, prop in sample_properties.iterrows():
            # Create a slightly modified version (should still find original)
            test_listing = {
                'city': prop['city'],
                'state': prop['state'],
                'bed': int(prop['bed']),
                'bath': int(prop['bath']),
                'house_size': int(prop['house_size']),
                'price': int(prop['price']) + 1000  # Slight price difference
            }

            matches, _ = self.engine.find_duplicates(test_listing, max_results=10)

            # Check if we found the original property
            found_original = any(
                match.bedrooms == prop['bed'] and
                match.bathrooms == prop['bath'] and
                match.house_size == prop['house_size'] and
                match.city.lower() == prop['city'].lower() and
                match.state.lower() == prop['state'].lower()
                for match in matches
            )

            recall_tests.append(found_original)

        recall_rate = sum(recall_tests) / len(recall_tests)
        self.test_results['recall_rate'] = recall_rate

        print(f"    Recall Rate: {recall_rate:.1%}")
        print(f"    Found {sum(recall_tests)}/{len(recall_tests)} exact duplicates")

        return recall_rate >= 1.0  # 100% requirement

    def test_200ms_performance(self) -> bool:
        """
        KPI Test 2: <200ms API response time
        Tests performance under various load conditions
        """
        print("\n Testing <200ms Performance Requirement...")

        # Test various property types for performance consistency
        test_cases = [
            # Urban properties (high density areas)
            {'city': 'San Juan', 'state': 'Puerto Rico', 'bed': 3, 'bath': 2, 'house_size': 1200, 'price': 250000},
            {'city': 'Ponce', 'state': 'Puerto Rico', 'bed': 4, 'bath': 3, 'house_size': 1800, 'price': 300000},

            # Rural properties (lower density)
            {'city': 'Adjuntas', 'state': 'Puerto Rico', 'bed': 3, 'bath': 2, 'house_size': 920, 'price': 105000},
            {'city': 'Ciales', 'state': 'Puerto Rico', 'bed': 2, 'bath': 1, 'house_size': 800, 'price': 80000},

            # Edge cases
            {'city': 'Unknown', 'state': 'Puerto Rico', 'bed': 10, 'bath': 8, 'house_size': 5000, 'price': 1000000},
            {'city': 'Mayaguez', 'state': 'Puerto Rico', 'bed': 1, 'bath': 1, 'house_size': 500, 'price': 50000},
        ]

        performance_tests = []
        for test_case in test_cases:
            # Run multiple times to get accurate timing
            times = []
            for _ in range(5):
                _, search_time = self.engine.find_duplicates(test_case)
                times.append(search_time)

            avg_time = statistics.mean(times)
            performance_tests.append(avg_time)

            print(f"    {test_case['city']}: {avg_time:.1f}ms (best: {min(times):.1f}ms)")

        max_time = max(performance_tests)
        avg_time = statistics.mean(performance_tests)

        self.test_results['max_response_time'] = max_time
        self.test_results['avg_response_time'] = avg_time

        print(f"    Average Response Time: {avg_time:.1f}ms")
        print(f"    Maximum Response Time: {max_time:.1f}ms")

        return max_time < 200  # <200ms requirement

    def test_confidence_scoring(self) -> bool:
        """
        KPI Test 3: Clear confidence scoring (0.95+ for obvious matches)
        Validates that confidence scores are meaningful
        """
        print("\n Testing Confidence Score Clarity...")

        # Test exact matches should have high confidence
        exact_match_test = {
            'city': 'Adjuntas',
            'state': 'Puerto Rico',
            'bed': 3,
            'bath': 2,
            'house_size': 920,
            'price': 105000
        }

        matches, _ = self.engine.find_duplicates(exact_match_test)

        high_confidence_matches = [m for m in matches if m.similarity_score >= 0.95]
        has_high_confidence = len(high_confidence_matches) > 0

        if matches:
            best_score = max(match.similarity_score for match in matches)
            self.test_results['best_confidence_score'] = best_score
            print(f"    Best Confidence Score: {best_score:.3f}")
            print(f"    High Confidence Matches (0.95): {len(high_confidence_matches)}")
        else:
            self.test_results['best_confidence_score'] = 0.0
            print("     No matches found for confidence testing")

        return has_high_confidence

    def test_load_handling(self) -> bool:
        """
        Additional Test: System stability under load
        Ensures consistent performance with multiple requests
        """
        print("\n Testing Load Handling...")

        # Simulate concurrent-like requests
        if self.engine.properties_df is None:
            raise RuntimeError("Properties dataframe is not loaded")
        test_properties = self.engine.properties_df.sample(20, random_state=123)

        load_test_times = []
        for _, prop in test_properties.iterrows():
            test_listing = {
                'city': prop['city'],
                'state': prop['state'],
                'bed': int(prop['bed']),
                'bath': int(prop['bath']),
                'house_size': int(prop['house_size']),
                'price': int(prop['price'])
            }

            _, search_time = self.engine.find_duplicates(test_listing)
            load_test_times.append(search_time)

        avg_load_time = statistics.mean(load_test_times)
        max_load_time = max(load_test_times)
        performance_consistent = max_load_time < 250  # Allow slight degradation under load

        self.test_results['load_test_avg'] = avg_load_time
        self.test_results['load_test_max'] = max_load_time

        print(f"    Average Time Under Load: {avg_load_time:.1f}ms")
        print(f"    Max Time Under Load: {max_load_time:.1f}ms")

        return performance_consistent

    def run_comprehensive_tests(self) -> Dict:
        """Run all KPI compliance tests and generate report"""
        print("="*60)
        print("PROPERTY SEARCH ENGINE - KPI COMPLIANCE TESTING")
        print("="*60)

        start_time = time.time()

        # Run all tests
        tests = {
            'recall_100_percent': self.test_100_percent_recall(),
            'performance_under_200ms': self.test_200ms_performance(),
            'confidence_scoring': self.test_confidence_scoring(),
            'load_handling': self.test_load_handling()
        }

        total_time = time.time() - start_time

        # Generate final report
        passed_tests = sum(tests.values())
        total_tests = len(tests)

        print("\n" + "="*60)
        print(" FINAL KPI COMPLIANCE REPORT")
        print("="*60)

        print(f"\n Tests Passed: {passed_tests}/{total_tests}")
        print(f"  Total Test Time: {total_time:.2f}s")

        print(f"\n Detailed Results:")
        for test_name, passed in tests.items():
            status = " PASS" if passed else " FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")

        # KPI Summary
        print(f"\n KPI Compliance Summary:")
        if 'recall_rate' in self.test_results:
            print(f"    Accuracy (Recall): {self.test_results['recall_rate']:.1%}")

        if 'avg_response_time' in self.test_results:
            print(f"    Performance: {self.test_results['avg_response_time']:.1f}ms avg")

        if 'best_confidence_score' in self.test_results:
            print(f"    Clarity: {self.test_results['best_confidence_score']:.3f} confidence")

        # Overall assessment
        kpi_compliant = tests['recall_100_percent'] and tests['performance_under_200ms']
        overall_status = " POC SUCCESS" if kpi_compliant else " NEEDS OPTIMIZATION"

        print(f"\n{overall_status}")
        print("="*60)

        return {
            'tests_passed': tests,
            'kpi_compliant': kpi_compliant,
            'detailed_results': self.test_results,
            'summary': {
                'passed': passed_tests,
                'total': total_tests,
                'test_time': total_time
            }
        }


def run_kpi_tests():
    """Main function to execute all KPI compliance tests"""
    try:
        tester = KPITestSuite('realtor_cleaned_final.csv')
        results = tester.run_comprehensive_tests()
        return results

    except FileNotFoundError:
        print(" Error: realtor_cleaned_final.csv not found!")
        print("   Please ensure the cleaned dataset exists in the current directory.")
        return None

    except Exception as e:
        print(f" Error during testing: {str(e)}")
        return None


if __name__ == "__main__":
    run_kpi_tests()
