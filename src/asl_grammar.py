"""
ASL Grammar Rules and Contextual Understanding.

This module implements ASL-specific grammar rules and provides
contextual interpretation of signed sentences.
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import re
import logging

logger = logging.getLogger(__name__)


class ASLGrammarRules:
    """
    Implements American Sign Language grammar rules.
    
    ASL grammar differs significantly from English:
    - Topic-Comment structure instead of Subject-Verb-Object
    - Temporal information often comes first
    - Questions indicated by facial expressions and word order
    - No articles (a, an, the) or copulas (is, are, was, were)
    - Directional verbs for spatial reference
    """
    
    def __init__(self):
        # Common ASL sentence structures
        self.topic_comment_indicators = ['ABOUT', 'REGARDING', 'CONCERNING']
        
        # Question words (WH-questions)
        self.wh_questions = ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW', 'WHICH']
        
        # Time indicators (typically come first)
        self.time_indicators = [
            'NOW', 'TODAY', 'TOMORROW', 'YESTERDAY',
            'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT',
            'PAST', 'FUTURE', 'BEFORE', 'AFTER',
            'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'
        ]
        
        # Articles and copulas to remove when translating to ASL
        self.english_articles = ['a', 'an', 'the']
        self.english_copulas = ['is', 'are', 'was', 'were', 'be', 'been', 'being']
        
        # Common ASL phrases
        self.common_phrases = {
            'HELLO': 'Hello / Hi',
            'GOODBYE': 'Goodbye / Bye',
            'THANK YOU': 'Thank you',
            'PLEASE': 'Please',
            'SORRY': 'I\'m sorry',
            'YES': 'Yes',
            'NO': 'No',
            'HELP': 'Help me / I need help',
            'UNDERSTAND': 'I understand / Do you understand?',
            'NOT UNDERSTAND': 'I don\'t understand'
        }
        
        # Directional verbs (meaning changes with direction)
        self.directional_verbs = [
            'GIVE', 'SEND', 'TELL', 'SHOW', 'ASK',
            'HELP', 'TEACH', 'PAY', 'INFORM'
        ]
    
    def is_question(self, signs: List[str]) -> bool:
        """
        Determine if a sequence of signs represents a question.
        
        In ASL, questions are indicated by:
        1. WH-question words (usually at the end for WH-questions)
        2. Raised eyebrows and head tilt for yes/no questions
        3. Word order changes
        """
        if not signs:
            return False
        
        # Check for WH-question words
        for sign in signs:
            if sign.upper() in self.wh_questions:
                return True
        
        return False
    
    def reorder_to_asl(self, english_words: List[str]) -> List[str]:
        """
        Reorder English words to ASL grammar structure.
        
        Args:
            english_words: List of English words
            
        Returns:
            Reordered list following ASL grammar
        """
        # Remove articles and copulas
        filtered = [
            word for word in english_words
            if word.lower() not in self.english_articles + self.english_copulas
        ]
        
        # Extract time indicators
        time_words = [w for w in filtered if w.upper() in self.time_indicators]
        other_words = [w for w in filtered if w.upper() not in self.time_indicators]
        
        # ASL order: TIME - TOPIC - COMMENT
        # Move time indicators to the beginning
        reordered = time_words + other_words
        
        return reordered
    
    def translate_to_asl_gloss(self, english_sentence: str) -> str:
        """
        Convert English sentence to ASL gloss notation.
        
        ASL gloss uses uppercase words to represent signs.
        
        Args:
            english_sentence: English sentence
            
        Returns:
            ASL gloss representation
        """
        # Remove punctuation
        sentence = re.sub(r'[^\w\s]', '', english_sentence)
        words = sentence.split()
        
        # Reorder to ASL grammar
        asl_words = self.reorder_to_asl(words)
        
        # Convert to uppercase (gloss notation)
        gloss = ' '.join([w.upper() for w in asl_words])
        
        return gloss
    
    def parse_asl_to_english(self, signs: List[str]) -> str:
        """
        Convert ASL signs to grammatically correct English.
        
        Args:
            signs: List of ASL signs (uppercase)
            
        Returns:
            English sentence
        """
        if not signs:
            return ""
        
        # Check for common phrases first
        sign_sequence = ' '.join(signs)
        if sign_sequence in self.common_phrases:
            return self.common_phrases[sign_sequence]
        
        # Convert signs to English words (lowercase)
        words = [sign.lower() for sign in signs]
        
        # Add appropriate articles and copulas
        english_sentence = self._add_english_grammar(words)
        
        # Capitalize first letter and add period
        if english_sentence:
            english_sentence = english_sentence[0].upper() + english_sentence[1:] + '.'
        
        return english_sentence
    
    def _add_english_grammar(self, words: List[str]) -> str:
        """
        Add English grammar elements (articles, copulas, etc.)
        
        This is a simplified version - full implementation would need
        more sophisticated NLP.
        """
        result = []
        
        for i, word in enumerate(words):
            # Add article before nouns (simplified heuristic)
            if i == 0 or (i > 0 and words[i-1] in ['see', 'want', 'need', 'have']):
                if word in ['apple', 'car', 'book', 'house']:
                    result.append('a')
            
            result.append(word)
            
            # Add copula for simple descriptions
            if i == 0 and word in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                next_word = words[i+1] if i+1 < len(words) else None
                if next_word and next_word in ['happy', 'sad', 'hungry', 'tired', 'good']:
                    result.append('am' if word == 'i' else 'are')
        
        return ' '.join(result)


class ContextManager:
    """
    Manages conversation context for better understanding.
    """
    
    def __init__(self, context_window: int = 5):
        self.context_window = context_window
        self.conversation_history = deque(maxlen=context_window)
        self.current_topic = None
        self.last_question = None
    
    def add_exchange(self, user_input: str, bot_response: str):
        """Add a conversation exchange to history"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'is_question': '?' in user_input
        })
        
        # Update topic tracking
        if '?' in user_input:
            self.last_question = user_input
    
    def get_context(self) -> str:
        """Get formatted context for LLM"""
        context_lines = []
        for exchange in self.conversation_history:
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Assistant: {exchange['bot']}")
        
        return '\n'.join(context_lines)
    
    def is_follow_up_question(self, signs: List[str]) -> bool:
        """
        Determine if current input is a follow-up question.
        """
        if not self.last_question:
            return False
        
        # Simple heuristic: short questions might be follow-ups
        return len(signs) <= 3 and any(s.upper() in ['WHAT', 'WHY', 'HOW'] for s in signs)
    
    def clear(self):
        """Clear conversation context"""
        self.conversation_history.clear()
        self.current_topic = None
        self.last_question = None


class PhrasePredictor:
    """
    Predicts and completes phrases based on context.
    """
    
    def __init__(self):
        # Common phrase completions
        self.phrase_patterns = {
            ('HELLO',): ['MY', 'NAME'],
            ('MY', 'NAME'): ['IS'],
            ('HOW', 'ARE'): ['YOU'],
            ('THANK',): ['YOU'],
            ('NICE', 'TO'): ['MEET', 'YOU'],
            ('WHERE', 'IS'): ['THE'],
            ('WHAT', 'IS'): ['YOUR'],
        }
        
        # Common ASL expressions
        self.expressions = {
            'greeting': ['HELLO', 'HI', 'GOOD MORNING', 'GOOD AFTERNOON'],
            'gratitude': ['THANK YOU', 'THANKS', 'APPRECIATE'],
            'apology': ['SORRY', 'EXCUSE ME', 'PARDON'],
            'agreement': ['YES', 'AGREE', 'OK', 'SURE'],
            'disagreement': ['NO', 'DISAGREE', 'NOT'],
        }
    
    def predict_next_signs(self, current_signs: List[str], top_k: int = 3) -> List[str]:
        """
        Predict the next most likely signs.
        
        Args:
            current_signs: Current sequence of signs
            top_k: Number of predictions to return
            
        Returns:
            List of predicted signs
        """
        if not current_signs:
            return []
        
        # Check last 1-3 signs for patterns
        for length in range(min(3, len(current_signs)), 0, -1):
            pattern = tuple(current_signs[-length:])
            if pattern in self.phrase_patterns:
                predictions = self.phrase_patterns[pattern]
                return predictions[:top_k]
        
        return []
    
    def complete_phrase(self, partial_signs: List[str]) -> Optional[List[str]]:
        """
        Complete a partial phrase if pattern is recognized.
        
        Args:
            partial_signs: Partial sequence of signs
            
        Returns:
            Completed phrase or None
        """
        for pattern, completion in self.phrase_patterns.items():
            if len(partial_signs) >= len(pattern):
                if tuple(partial_signs[-len(pattern):]) == pattern:
                    return list(pattern) + completion
        
        return None
    
    def detect_expression_type(self, signs: List[str]) -> Optional[str]:
        """
        Detect the type of expression (greeting, gratitude, etc.)
        
        Args:
            signs: Sequence of signs
            
        Returns:
            Expression type or None
        """
        sign_sequence = ' '.join(signs)
        
        for expr_type, patterns in self.expressions.items():
            for pattern in patterns:
                if pattern in sign_sequence:
                    return expr_type
        
        return None


class ASLInterpreter:
    """
    Main interpreter combining grammar rules, context, and prediction.
    """
    
    def __init__(self):
        self.grammar = ASLGrammarRules()
        self.context = ContextManager()
        self.predictor = PhrasePredictor()
    
    def interpret_signs(self, signs: List[str]) -> Dict:
        """
        Interpret a sequence of signs with full context.
        
        Args:
            signs: List of ASL signs
            
        Returns:
            Dictionary with interpretation results
        """
        result = {
            'signs': signs,
            'english': '',
            'is_question': False,
            'expression_type': None,
            'predicted_next': [],
            'completed_phrase': None
        }
        
        # Detect expression type
        result['expression_type'] = self.predictor.detect_expression_type(signs)
        
        # Check if it's a question
        result['is_question'] = self.grammar.is_question(signs)
        
        # Translate to English
        result['english'] = self.grammar.parse_asl_to_english(signs)
        
        # Predict next signs
        result['predicted_next'] = self.predictor.predict_next_signs(signs)
        
        # Try to complete phrase
        result['completed_phrase'] = self.predictor.complete_phrase(signs)
        
        return result
    
    def add_to_context(self, signs: List[str], response: str):
        """Add interpreted exchange to context"""
        english = self.grammar.parse_asl_to_english(signs)
        self.context.add_exchange(english, response)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    interpreter = ASLInterpreter()
    
    # Test cases
    test_sequences = [
        ['HELLO'],
        ['WHAT', 'YOUR', 'NAME'],
        ['I', 'HUNGRY'],
        ['TOMORROW', 'I', 'GO', 'SCHOOL'],
        ['THANK', 'YOU'],
    ]
    
    print("ASL Grammar Interpreter Examples:\n")
    for signs in test_sequences:
        result = interpreter.interpret_signs(signs)
        print(f"Signs: {' '.join(signs)}")
        print(f"English: {result['english']}")
        print(f"Question: {result['is_question']}")
        print(f"Type: {result['expression_type']}")
        print()
