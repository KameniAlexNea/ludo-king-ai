"""Behavior classification for game segments."""

from typing import Dict

from .models import BehaviorCharacteristics


class BehaviorClassifier:
    """Classifies behavior characteristics into strategy styles."""
    
    # Style definitions based on characteristic thresholds
    STYLE_PROFILES: Dict[str, Dict[str, tuple]] = {
        "aggressive": {
            "aggression": (0.6, 1.0),
            "risk_taking": (0.5, 1.0),
            "defensiveness": (0.0, 0.4),
        },
        "defensive": {
            "aggression": (0.0, 0.4),
            "defensiveness": (0.6, 1.0),
            "risk_taking": (0.0, 0.3),
        },
        "explorer": {
            "exploration": (0.75, 1.0),
            "finishing": (0.0, 0.4),
        },
        "finisher": {
            "finishing": (0.6, 1.0),
            "exploration": (0.0, 0.5),
        },
        "blockader": {
            "blockade_usage": (0.6, 1.0),
            "defensiveness": (0.4, 1.0),
        },
        "opportunist": {
            "aggression": (0.4, 0.7),
            "risk_taking": (0.3, 0.6),
            "blockade_usage": (0.3, 0.7),
        },
        "balanced": {
            # Everything in moderate range
            "aggression": (0.3, 0.7),
            "defensiveness": (0.3, 0.7),
            "exploration": (0.4, 0.8),
        },
    }
    
    def classify(
        self,
        characteristics: BehaviorCharacteristics,
    ) -> tuple[str, float]:
        """
        Classify behavior into a strategy style.
        
        Parameters
        ----------
        characteristics : BehaviorCharacteristics
            Computed behavioral characteristics.
            
        Returns
        -------
        tuple[str, float]
            (style_name, confidence) where confidence is 0-1.
        """
        scores = {}
        
        for style, profile in self.STYLE_PROFILES.items():
            score = self._match_score(characteristics, profile)
            scores[style] = score
        
        # Find best match
        best_style = max(scores, key=scores.get)
        confidence = scores[best_style]
        
        return best_style, confidence
    
    def _match_score(
        self,
        characteristics: BehaviorCharacteristics,
        profile: Dict[str, tuple],
    ) -> float:
        """
        Compute how well characteristics match a style profile.
        
        Parameters
        ----------
        characteristics : BehaviorCharacteristics
            Observed characteristics.
        profile : Dict[str, tuple]
            Style profile with (min, max) ranges for each characteristic.
            
        Returns
        -------
        float
            Match score 0-1.
        """
        matches = []
        
        for char_name, (min_val, max_val) in profile.items():
            char_value = getattr(characteristics, char_name)
            
            # Check if value is in range
            if min_val <= char_value <= max_val:
                # Perfect match in center of range
                center = (min_val + max_val) / 2
                distance = abs(char_value - center)
                range_size = (max_val - min_val) / 2
                match_score = 1.0 - (distance / range_size) if range_size > 0 else 1.0
                matches.append(match_score)
            else:
                # Out of range - penalize based on distance
                if char_value < min_val:
                    distance = min_val - char_value
                else:
                    distance = char_value - max_val
                matches.append(max(0.0, 1.0 - distance))
        
        # Average match across all characteristics
        return sum(matches) / len(matches) if matches else 0.0
    
    def classify_with_context(
        self,
        characteristics: BehaviorCharacteristics,
        phase: str = "midgame",
    ) -> tuple[str, float]:
        """
        Classify with game phase context.
        
        Different phases have different natural behaviors:
        - Opening: more exploration
        - Midgame: more aggression/blocking
        - Endgame: more finishing focus
        
        Parameters
        ----------
        characteristics : BehaviorCharacteristics
            Computed characteristics.
        phase : str
            Game phase: "opening", "midgame", or "endgame".
            
        Returns
        -------
        tuple[str, float]
            (style_name, confidence).
        """
        # Adjust expectations based on phase
        if phase == "opening":
            # In opening, high exploration is normal
            if characteristics.exploration > 0.7:
                return "explorer", 0.9
        elif phase == "endgame":
            # In endgame, finishing focus is normal
            if characteristics.finishing > 0.6:
                return "finisher", 0.9
        
        # Otherwise use standard classification
        return self.classify(characteristics)
