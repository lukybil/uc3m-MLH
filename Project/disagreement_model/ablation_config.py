"""
Configuration for ablation study experiments.

Defines semantic feature groups and ablation experiment configurations.
Extensible design allows adding data-driven correlation groups later.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class FeatureGroup:
    """Represents a group of related features for ablation analysis"""

    name: str
    description: str
    features: List[str]
    category: str  # 'semantic', 'data_driven', 'strategic'


class AblationConfig:
    """Central configuration for all ablation study experiments"""

    def __init__(self):
        """Initialize with semantic feature groups"""
        self.semantic_groups = self._define_semantic_groups()
        self.strategic_experiments = self._define_strategic_experiments()

        # Placeholder for data-driven groups (to be added later)
        self.data_driven_groups = {}

    def _define_semantic_groups(self) -> Dict[str, FeatureGroup]:
        """Define semantic feature groups based on clinical/administrative meaning"""

        groups = {}

        # ===== CORE CLINICAL GROUPS =====

        # Group 1: Depression (PHQ-9)
        groups["depression"] = FeatureGroup(
            name="depression",
            description="Depression assessment (PHQ-9) - total scores, severity, individual items",
            category="semantic",
            features=[
                # Total scores and severity
                "phq9_total_score",
                "phq9_severity_group_code",
                "phq9_severity_group_label",
                "phq9_computed_total",
                # Individual items
                "phq9_1",
                "phq9_2",
                "phq9_3",
                "phq9_4",
                "phq9_5",
                "phq9_6",
                "phq9_7",
                "phq9_8",
                "phq9_9",
            ],
        )

        # Group 2: Anxiety (GAD-7)
        groups["anxiety"] = FeatureGroup(
            name="anxiety",
            description="Anxiety assessment (GAD-7) - total scores, severity, individual items",
            category="semantic",
            features=[
                # Total scores and severity
                "gad7_total_score",
                "gad7_severity_group_code",
                "gad7_severity_group_label",
                "gad7_computed_total",
                # Individual items
                "gad7_1",
                "gad7_2",
                "gad7_3",
                "gad7_4",
                "gad7_5",
                "gad7_6",
                "gad7_7",
            ],
        )

        # Group 3: Suicide Risk (C-SSRS)
        groups["suicide_risk"] = FeatureGroup(
            name="suicide_risk",
            description="Suicide risk assessment (C-SSRS) - total scores, groups, individual items",
            category="semantic",
            features=[
                # Total scores
                "cssrs_total_score_1",
                "cssrs_total_score_2",
                "cssrs_total_score_3",
                # Groups
                "cssrs_group_2a_code",
                "cssrs_group_2a_label",
                "cssrs_group_2b_code",
                "cssrs_group_2b_label",
                "cssrs_group_2c_code",
                "cssrs_group_2c_label",
                "cssrs_group_2d_code",
                "cssrs_group_2d_label",
                # Individual items
                "cssrs_wish_to_be_dead",
                "cssrs_non_specific_active_suicidal_ideation",
                "cssrs_active_ideation_with_plan_or_methods",
                "cssrs_active_ideation_with_intent",
                "cssrs_active_ideation_with_specific_plan",
                "cssrs_suicidal_behavior",
                "cssrs_suicidal_behavior_last_3_months",
                "cssrs_result_1a",
                "cssrs_result_1b",
                "cssrs_result_1c",
                "cssrs_result_1d",
            ],
        )

        # Group 4: Substance Use (AUDIT-C + DAST)
        groups["substance_use"] = FeatureGroup(
            name="substance_use",
            description="Substance use assessment (AUDIT-C alcohol + DAST drugs)",
            category="semantic",
            features=[
                # AUDIT-C (alcohol)
                "auditc_total_score",
                "auditc_risk_group_code",
                "auditc_risk_group_label",
                "auditc_1_frequency_of_drinking",
                "auditc_2_drinks_on_typical_drinking_day",
                "auditc_3_frequency_six_or_more_drinks",
                # DAST (drugs)
                "dast_total_score",
                "dast_1",
                "dast_2",
                "dast_3",
                "dast_4",
            ],
        )

        # Group 5: Psychosis Symptoms
        groups["psychosis"] = FeatureGroup(
            name="psychosis",
            description="Psychosis symptoms screening (4 items + count)",
            category="semantic",
            features=[
                "psychosis_1_thoughts_being_interfered_with_or_controlled",
                "psychosis_2_group_conspiring_to_harm_you",
                "psychosis_3_something_very_strange_happening",
                "psychosis_4_heard_voices_with_no_one_there",
                "psychosis_symptom_count",
            ],
        )

        # Group 6: Well-being & Other Clinical
        groups["wellbeing_other"] = FeatureGroup(
            name="wellbeing_other",
            description="Well-being (WHO-5), self-harm, medication, smoking",
            category="semantic",
            features=[
                # WHO-5 well-being
                "who5_total_score",
                "who5_cheerful",
                "who5_calm",
                "who5_active",
                "who5_fresh",
                "who5_daily_life",
                # Self-harm
                "nssi_ever_intentional_self_injury_without_suicidal_intent",
                "nssi_last_month_frequency",
                # Medication & smoking
                "current_psychotropic_medication",
                "current_smoker",
                "cigarettes_per_day",
            ],
        )

        # ===== ADMINISTRATIVE & CONTEXTUAL GROUPS =====

        # Group 7: Administrative/Service Features
        groups["administrative"] = FeatureGroup(
            name="administrative",
            description="Service, center, evaluating user, patient ID, care circuit",
            category="semantic",
            features=[
                "center_id",
                "center_id_2",
                "patient_id_2",
                "service",
                "care_circuit",
                "response_type",
                "evaluating_user",
                "requesting_service",
                "requesting_service_2",
                "form_status",
            ],
        )

        # Group 8: Demographics
        groups["demographics"] = FeatureGroup(
            name="demographics",
            description="Age, sex, location of birth, pregnancy status",
            category="semantic",
            features=[
                "age",
                "sex",
                "sex_2",
                "location_of_birth",
                "location_of_birth_2",
                "date_of_birth_2",
                "is_pregnant_woman",
            ],
        )

        # Group 9: Temporal Features
        groups["temporal"] = FeatureGroup(
            name="temporal",
            description="Time delays between request, assessment, evaluation, publication",
            category="semantic",
            features=[
                "days_request_to_assessment",
                "days_assessment_to_evaluation",
                "days_request_to_publication",
                "completion_date",
                "results_date",
                "evaluation_date",
            ],
        )

        # Group 10: Personality Traits (BFI-10)
        groups["personality"] = FeatureGroup(
            name="personality",
            description="Big Five personality traits (BFI-10) - items and computed scores",
            category="semantic",
            features=[
                # Individual items
                "bfi10_extraverted_enthusiastic",
                "bfi10_critical_quarrelsome",
                "bfi10_dependable_self_disciplined",
                "bfi10_anxious_easily_upset",
                "bfi10_open_to_new_experiences_and_ideas",
                "bfi10_reserved_quiet",
                "bfi10_sympathetic_warm",
                "bfi10_disorganized_careless",
                "bfi10_calm_emotionally_stable",
                "bfi10_prefers_traditional_less_creative",
                # Computed scores
                "bfi10_total_score",
                "bfi10_extraversion_score",
                "bfi10_agreeableness_score",
                "bfi10_conscientiousness_score",
                "bfi10_emotional_stability_score",
                "bfi10_openness_to_experience_score",
            ],
        )

        # Group 11: Engineered Composite Features
        groups["engineered_composites"] = FeatureGroup(
            name="engineered_composites",
            description="Engineered features combining multiple sources",
            category="semantic",
            features=[
                "combined_depression_anxiety_score",
                "phq9_computed_total",
                "gad7_computed_total",
                "psychosis_symptom_count",
            ],
        )

        return groups

    def _define_strategic_experiments(self) -> Dict[str, Dict]:
        """Define strategic ablation experiments (combinations of groups)"""

        experiments = {}

        # Experiment 1: Clinical Only (remove admin, demo, temporal)
        experiments["clinical_only"] = {
            "description": "Only clinical features (remove administrative, demographics, temporal)",
            "include_groups": [
                "depression",
                "anxiety",
                "suicide_risk",
                "substance_use",
                "psychosis",
                "wellbeing_other",
                "personality",
                "engineered_composites",
            ],
            "exclude_groups": ["administrative", "demographics", "temporal"],
        }

        # Experiment 2: No Administrative (potentially leaky features)
        experiments["no_administrative"] = {
            "description": "Remove administrative features (service, evaluating_user, patient_id)",
            "include_groups": None,  # All except excluded
            "exclude_groups": ["administrative"],
        }

        # Experiment 3: No Temporal
        experiments["no_temporal"] = {
            "description": "Remove temporal features (time delays)",
            "include_groups": None,
            "exclude_groups": ["temporal"],
        }

        # Experiment 4: Scores Only (remove individual items)
        experiments["scores_only"] = {
            "description": "Only total scores and summary features (no individual items)",
            "include_groups": None,
            "exclude_groups": [],
            "custom_filter": "scores_only",  # Special handling needed
        }

        # Experiment 5: Core Clinical Only (Depression + Anxiety + Suicide Risk)
        experiments["core_clinical"] = {
            "description": "Only core mental health assessments (PHQ-9, GAD-7, C-SSRS)",
            "include_groups": ["depression", "anxiety", "suicide_risk"],
            "exclude_groups": [],
        }

        # Experiment 6: No Personality
        experiments["no_personality"] = {
            "description": "Remove personality traits (BFI-10)",
            "include_groups": None,
            "exclude_groups": ["personality"],
        }

        return experiments

    def get_all_features_in_group(self, group_name: str) -> List[str]:
        """Get all feature names in a semantic group"""
        if group_name in self.semantic_groups:
            return self.semantic_groups[group_name].features
        elif group_name in self.data_driven_groups:
            return self.data_driven_groups[group_name].features
        else:
            raise ValueError(f"Unknown feature group: {group_name}")

    def get_all_features_in_groups(self, group_names: List[str]) -> Set[str]:
        """Get union of all features in specified groups"""
        all_features = set()
        for group_name in group_names:
            all_features.update(self.get_all_features_in_group(group_name))
        return all_features

    def get_features_for_experiment(
        self, experiment_name: str, all_available_features: List[str]
    ) -> List[str]:
        """
        Get feature list for a specific ablation experiment.

        Args:
            experiment_name: Name of the experiment ('baseline' or key from strategic_experiments)
            all_available_features: Complete list of available features in dataset

        Returns:
            List of features to use in this experiment
        """
        if experiment_name == "baseline":
            return all_available_features

        if experiment_name in self.strategic_experiments:
            exp_config = self.strategic_experiments[experiment_name]

            # Handle custom filters
            if exp_config.get("custom_filter") == "scores_only":
                return self._filter_scores_only(all_available_features)

            # Handle include/exclude logic
            if exp_config["include_groups"] is not None:
                # Only include specified groups
                return list(
                    self.get_all_features_in_groups(exp_config["include_groups"])
                )
            else:
                # Include all except excluded groups
                excluded_features = self.get_all_features_in_groups(
                    exp_config["exclude_groups"]
                )
                return [f for f in all_available_features if f not in excluded_features]

        # If it's a single group ablation (leave-one-out)
        if experiment_name.startswith("without_"):
            group_to_remove = experiment_name.replace("without_", "")
            if (
                group_to_remove in self.semantic_groups
                or group_to_remove in self.data_driven_groups
            ):
                excluded_features = set(self.get_all_features_in_group(group_to_remove))
                return [f for f in all_available_features if f not in excluded_features]

        raise ValueError(f"Unknown experiment: {experiment_name}")

    def _filter_scores_only(self, all_features: List[str]) -> List[str]:
        """Filter to keep only summary scores, removing individual items"""
        # Keep features that are:
        # - total_score, severity, risk_group, computed_total
        # - NOT individual items (phq9_1, gad7_3, etc.)
        scores_only = []

        item_patterns = [
            "_1",
            "_2",
            "_3",
            "_4",
            "_5",
            "_6",
            "_7",
            "_8",
            "_9",
            "phq9_1",
            "gad7_",
            "who5_cheerful",
            "who5_calm",
            "who5_active",
            "who5_fresh",
            "who5_daily_life",
            "cssrs_wish",
            "cssrs_non_specific",
            "cssrs_active_ideation",
            "cssrs_suicidal_behavior",
            "cssrs_result",
            "auditc_1_",
            "auditc_2_",
            "auditc_3_",
            "dast_1",
            "dast_2",
            "dast_3",
            "dast_4",
            "bfi10_extraverted",
            "bfi10_critical",
            "bfi10_dependable",
            "bfi10_anxious",
            "bfi10_open_to",
            "bfi10_reserved",
            "bfi10_sympathetic",
            "bfi10_disorganized",
            "bfi10_calm",
            "bfi10_prefers",
        ]

        for feature in all_features:
            # Check if it's an individual item
            is_item = any(pattern in feature for pattern in item_patterns)
            if not is_item:
                scores_only.append(feature)

        return scores_only

    def get_leave_one_out_experiments(self) -> List[str]:
        """Get list of leave-one-out experiment names for each semantic group"""
        return [f"without_{group_name}" for group_name in self.semantic_groups.keys()]

    def add_data_driven_group(self, group: FeatureGroup):
        """Add a data-driven feature group (e.g., from correlation analysis)"""
        if group.category != "data_driven":
            raise ValueError("Group category must be 'data_driven'")
        self.data_driven_groups[group.name] = group

    def get_all_experiment_names(self, include_leave_one_out: bool = True) -> List[str]:
        """Get all available experiment names"""
        experiments = ["baseline"] + list(self.strategic_experiments.keys())
        if include_leave_one_out:
            experiments.extend(self.get_leave_one_out_experiments())
        return experiments

    def print_summary(self):
        """Print summary of all available feature groups and experiments"""
        print("=" * 80)
        print("ABLATION STUDY CONFIGURATION SUMMARY")
        print("=" * 80)

        print(f"\nSEMANTIC FEATURE GROUPS ({len(self.semantic_groups)}):")
        print("-" * 80)
        for name, group in self.semantic_groups.items():
            print(f"\n{name.upper()}: {group.description}")
            print(f"  Features: {len(group.features)}")
            print(f"  Examples: {', '.join(group.features[:3])}...")

        if self.data_driven_groups:
            print(f"\n\nDATA-DRIVEN FEATURE GROUPS ({len(self.data_driven_groups)}):")
            print("-" * 80)
            for name, group in self.data_driven_groups.items():
                print(f"\n{name.upper()}: {group.description}")
                print(f"  Features: {len(group.features)}")

        print(f"\n\nSTRATEGIC EXPERIMENTS ({len(self.strategic_experiments)}):")
        print("-" * 80)
        for name, exp in self.strategic_experiments.items():
            print(f"\n{name}: {exp['description']}")
            if exp["include_groups"]:
                print(f"  Include groups: {', '.join(exp['include_groups'])}")
            if exp["exclude_groups"]:
                print(f"  Exclude groups: {', '.join(exp['exclude_groups'])}")

        print(f"\n\nLEAVE-ONE-OUT EXPERIMENTS ({len(self.semantic_groups)}):")
        print("-" * 80)
        for exp_name in self.get_leave_one_out_experiments():
            group_name = exp_name.replace("without_", "")
            print(f"  {exp_name}")

        print("\n" + "=" * 80)


# Convenience function
def get_default_ablation_config() -> AblationConfig:
    """Get default ablation configuration"""
    return AblationConfig()


if __name__ == "__main__":
    # Demo
    config = AblationConfig()
    config.print_summary()
