import sys
import cv2
import numpy as np
import requests
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QScrollArea,
                              QComboBox, QGroupBox, QFrame, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
import os
import urllib.request
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from collections import deque
import time
import logging

# Try to import enhanced modules (optional for backward compatibility)
try:
    from src.asl_grammar import ASLInterpreter
    from src.utils import load_config
    ENHANCED_MODE = True
except ImportError:
    ENHANCED_MODE = False
    logging.warning("Enhanced modules not available. Running in basic mode.")

# Setup logging
logging.basicConfig(level=logging.INFO)

class ASLRecognizer:
    """Handles ASL recognition using MediaPipe hand tracking with enhanced sentence-level support"""
    def __init__(self, config=None):
        # Load configuration
        self.config = config or {}
        self.enhanced_mode = ENHANCED_MODE
        
        # Try to find a local model bundle, otherwise download the default
        self.model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(self.model_path):
            try:
                url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
                urllib.request.urlretrieve(url, self.model_path)
            except Exception:
                raise RuntimeError(
                    "Hand landmarker model not found and automatic download failed.\n"
                    "Place 'hand_landmarker.task' next to chat.py or install a mediapipe wheel that provides the Tasks API."
                )

        try:
            self.landmarker = HandLandmarker.create_from_model_path(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create HandLandmarker: {e}")

        # Buffer for gesture detection
        recognition_config = self.config.get('recognition', {})
        buffer_size = recognition_config.get('gesture_buffer_size', 20)
        self.gesture_buffer = deque(maxlen=buffer_size)
        self.word_buffer = []
        self.sign_sequence = []  # For enhanced sentence tracking
        self.last_gesture_time = time.time()
        self.last_complete_time = time.time()
        self.sentence_complete = False
        
        # Enhanced features
        if self.enhanced_mode:
            self.interpreter = ASLInterpreter()
            logging.info("ASL Vision Bot running in ENHANCED mode with sentence-level recognition")
        else:
            self.interpreter = None
            logging.info("ASL Vision Bot running in BASIC mode")
        
    def process_frame(self, frame):
        """Process frame and detect hand gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to MediaPipe Image and run detection
        mp_image = Image(ImageFormat.SRGB, rgb_frame)
        try:
            results = self.landmarker.detect(mp_image)
        except Exception:
            results = None

        detected_gesture = None

        if results and getattr(results, 'hand_landmarks', None):
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks and connections onto the frame
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)

                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    start = hand_landmarks[conn.start]
                    end = hand_landmarks[conn.end]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

                # Analyze hand pose
                detected_gesture = self._analyze_hand_pose(hand_landmarks)
        
        # Update gesture buffer
        if detected_gesture and detected_gesture != 'NONE':
            self.gesture_buffer.append(detected_gesture)
            
            # Check if gesture is stable
            if len(self.gesture_buffer) >= 12:
                most_common = max(set(self.gesture_buffer), 
                                key=list(self.gesture_buffer).count)
                
                # Add to word if gesture held for sufficient time
                current_time = time.time()
                if current_time - self.last_gesture_time > 1.0:
                    if most_common == 'SPACE':
                        self.word_buffer.append(' ')
                        # Track sign sequence for enhanced mode
                        if self.sign_sequence:
                            # Space indicates word boundary
                            pass
                    elif most_common == 'DELETE' and self.word_buffer:
                        self.word_buffer.pop()
                        if self.sign_sequence:
                            self.sign_sequence.pop()
                    elif most_common == 'SUBMIT':
                        self.sentence_complete = True
                        self.last_complete_time = current_time
                    elif most_common != 'NONE':
                        self.word_buffer.append(most_common)
                        # Add to sign sequence for enhanced interpretation
                        self.sign_sequence.append(most_common)
                    
                    self.last_gesture_time = current_time
                    self.gesture_buffer.clear()
        else:
            # Check for sentence timeout (3 seconds of no gestures)
            current_time = time.time()
            if len(self.word_buffer) > 0 and current_time - self.last_gesture_time > 3.0:
                if not self.sentence_complete:
                    self.sentence_complete = True
                    self.last_complete_time = current_time
        
        current_word = ''.join(self.word_buffer)
        return frame, detected_gesture, current_word
    
    def _analyze_hand_pose(self, landmarks):
        """Analyze hand landmarks (list of NormalizedLandmark) to recognize gestures"""
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Finger states
        fingers_closed = all([
            index_tip.y > landmarks[6].y,
            middle_tip.y > landmarks[10].y,
            ring_tip.y > landmarks[14].y,
            pinky_tip.y > landmarks[18].y,
        ])

        fingers_open = all([
            index_tip.y < landmarks[6].y,
            middle_tip.y < landmarks[10].y,
            ring_tip.y < landmarks[14].y,
            pinky_tip.y < landmarks[18].y,
        ])

        # Specific gestures
        l_shape = (
            index_tip.y < landmarks[6].y
            and abs(thumb_tip.x - wrist.x) > 0.08
            and middle_tip.y > landmarks[10].y
        )

        o_shape = (
            abs(thumb_tip.y - index_tip.y) < 0.05
            and abs(thumb_tip.x - index_tip.x) < 0.05
        )

        thumbs_up = (thumb_tip.y < wrist.y - 0.1 and fingers_closed)

        palm_down = (
            fingers_open
            and all([landmarks[i].y > wrist.y for i in [8, 12, 16, 20]])
            and abs(index_tip.y - pinky_tip.y) < 0.05
        )

        if thumbs_up:
            return 'SUBMIT'
        elif palm_down:
            return 'SPACE'
        elif l_shape:
            return 'L'
        elif o_shape:
            return 'O'
        elif fingers_closed and abs(thumb_tip.x - index_tip.x) < 0.05:
            return 'A'
        elif fingers_open and thumb_tip.y < index_tip.y:
            return 'B'
        elif abs(index_tip.y - middle_tip.y) < 0.03 and ring_tip.y > landmarks[14].y:
            return 'V'
        else:
            return 'NONE'
    
    def is_sentence_complete(self):
        """Check if sentence is complete"""
        if self.sentence_complete:
            self.sentence_complete = False
            return True
        return False
    
    def get_and_clear_text(self):
        """Get current text and clear buffer"""
        text = ''.join(self.word_buffer)
        
        # Enhanced interpretation if available
        if self.enhanced_mode and self.interpreter and self.sign_sequence:
            try:
                interpretation = self.interpreter.interpret_signs(self.sign_sequence)
                if interpretation.get('english'):
                    # Use grammatically correct English translation
                    text = interpretation['english']
                    logging.info(f"Enhanced interpretation: {self.sign_sequence} -> {text}")
            except Exception as e:
                logging.warning(f"Enhanced interpretation failed: {e}")
        
        # Clear buffers
        self.word_buffer.clear()
        self.sign_sequence.clear()
        return text
    
    def get_sign_sequence(self):
        """Get the current sign sequence for enhanced processing"""
        return self.sign_sequence.copy()
    
    def clear_word(self):
        """Clear the current word buffer"""
        self.word_buffer.clear()
        self.sign_sequence.clear()

class CameraThread(QThread):
    """Thread for camera capture"""
    frame_ready = pyqtSignal(np.ndarray, str, str)
    sentence_complete = pyqtSignal(str)
    
    def __init__(self, camera_index=0, config=None):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.asl_recognizer = ASLRecognizer(config=config)
        self.frame_count = 0
        self.process_interval = 3  # Process every 3 frames
    
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                
                # Process every N frames
                if self.frame_count % self.process_interval == 0:
                    processed_frame, gesture, word = self.asl_recognizer.process_frame(frame)
                    
                    # Emit frame with recognition results
                    self.frame_ready.emit(processed_frame, 
                                        gesture if gesture else "None", 
                                        word)
                    
                    # Check if sentence is complete
                    if self.asl_recognizer.is_sentence_complete():
                        complete_text = self.asl_recognizer.get_and_clear_text()
                        if complete_text.strip():
                            self.sentence_complete.emit(complete_text)
        
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class LLMHandler:
    """Handles communication with local Ollama LLM with enhanced ASL context awareness"""
    def __init__(self, model="llama3.2:1b", config=None):
        self.config = config or {}
        llm_config = self.config.get('llm', {})
        
        self.model = llm_config.get('model', model)
        self.base_url = llm_config.get('base_url', "http://localhost:11434")
        self.timeout = llm_config.get('timeout', 30)
        
        # Context management
        self.context_window = llm_config.get('context_window', 5)
        self.conversation_history = []
        
        # Enhanced mode check
        self.enhanced_mode = ENHANCED_MODE
        if self.enhanced_mode:
            try:
                from src.asl_grammar import ContextManager
                self.context_manager = ContextManager(self.context_window)
                logging.info("LLM handler initialized with enhanced context management")
            except ImportError:
                self.context_manager = None
                logging.warning("Context manager not available")
        else:
            self.context_manager = None
    
    def generate_response(self, prompt):
        """Generate response from LLM"""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_query_intent(self, query):
        """Determine if query needs web search"""
        prompt = f"""Analyze this question and determine if it requires current information from the internet or can be answered with general knowledge.

Question: {query}

Respond with ONLY one word:
- "SEARCH" if it needs current/real-time information (weather, news, prices, current events, latest info)
- "ANSWER" if it can be answered with general knowledge

Response:"""
        
        response = self.generate_response(prompt).strip().upper()
        return "SEARCH" in response
    
    def respond_to_query(self, query, context=None):
        """Generate conversational response with optional context"""
        # Build context-aware prompt
        if context and self.enhanced_mode and self.context_manager:
            context_text = self.context_manager.get_context()
            prompt = f"""You are a helpful AI assistant having a conversation through sign language. 

Previous conversation:
{context_text}

Current message (translated from ASL): {query}

Respond naturally and concisely (2-3 sentences). Be aware that the input comes from sign language translation."""
        else:
            prompt = f"""You are a helpful AI assistant having a conversation through sign language. Respond naturally and concisely to this message: {query}

Keep your response brief (2-3 sentences) and conversational."""
        
        response = self.generate_response(prompt)
        
        # Update context if available
        if self.enhanced_mode and self.context_manager:
            self.context_manager.add_exchange(query, response)
        
        return response
    
    def search_and_respond(self, query, search_results):
        """Generate response based on search results"""
        prompt = f"""Based on the following search results, provide a clear and concise answer to the question: {query}

Search Results:
{search_results}

Provide a brief, helpful answer (3-4 sentences):"""
        
        return self.generate_response(prompt)
    
    def web_search(self, query):
        """Perform web search using DuckDuckGo API"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if data.get('Abstract'):
                    results.append(f"Summary: {data['Abstract']}")
                
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:3]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append(f"• {topic['Text']}")
                
                return '\n'.join(results) if results else "No specific results found, but I can provide general information."
            
            return "Search service unavailable"
        
        except Exception as e:
            return f"Search error: {str(e)}"

class MessageWidget(QFrame):
    """Widget to display a single message"""
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.init_ui(text, is_user)
    
    def init_ui(self, text, is_user):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        message_label.setFont(QFont("Arial", 11))
        
        if is_user:
            # User message - right aligned, blue background
            message_label.setStyleSheet("""
                background-color: #0084ff;
                color: white;
                padding: 10px 15px;
                border-radius: 18px;
                max-width: 400px;
            """)
            layout.addStretch()
            layout.addWidget(message_label)
        else:
            # Bot message - left aligned, gray background
            message_label.setStyleSheet("""
                background-color: #e4e6eb;
                color: black;
                padding: 10px 15px;
                border-radius: 18px;
                max-width: 400px;
            """)
            layout.addWidget(message_label)
            layout.addStretch()

class SearchButtonWidget(QFrame):
    """Widget with search button"""
    search_clicked = pyqtSignal(str)
    
    def __init__(self, query, parent=None):
        super().__init__(parent)
        self.query = query
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        search_btn = QPushButton(f"🔍 Search the web for: '{self.query[:50]}...'")
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        search_btn.clicked.connect(lambda: self.search_clicked.emit(self.query))
        
        layout.addWidget(search_btn)
        layout.addStretch()

class ASLChatbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load configuration if available
        try:
            if ENHANCED_MODE:
                from src.utils import load_config
                self.config = load_config('config.yaml')
                logging.info("Configuration loaded successfully")
            else:
                self.config = {}
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            self.config = {}
        
        self.camera_thread = None
        self.llm_handler = LLMHandler(config=self.config)
        self.current_query = ""
        self.init_ui()
        
        # Auto-start camera
        QTimer.singleShot(500, self.start_camera)
    
    def init_ui(self):
        # Set window title with mode indicator
        mode_text = "Enhanced" if ENHANCED_MODE else "Basic"
        self.setWindowTitle(f"ASL Sign Language Chatbot with LLM ({mode_text} Mode)")
        self.setGeometry(100, 100, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left side - Camera feed
        left_layout = QVBoxLayout()
        
        # Camera controls
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Camera 0", "Camera 1", "Camera 2"])
        controls_layout.addWidget(QLabel("Camera:"))
        controls_layout.addWidget(self.camera_selector)
        
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        camera_layout.addLayout(controls_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_layout.addWidget(self.video_label)
        
        # Recognition status
        status_layout = QVBoxLayout()
        self.gesture_label = QLabel("Gesture: None")
        self.gesture_label.setFont(QFont("Arial", 11))
        status_layout.addWidget(self.gesture_label)
        
        self.word_label = QLabel("Signing: ")
        self.word_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self.word_label.setStyleSheet("color: #0084ff; padding: 5px;")
        self.word_label.setWordWrap(True)
        status_layout.addWidget(self.word_label)
        
        help_label = QLabel("💡 Hold gestures for 1s | 3s pause = auto-submit | Thumbs up = submit")
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(help_label)
        
        camera_layout.addLayout(status_layout)
        camera_group.setLayout(camera_layout)
        left_layout.addWidget(camera_group)
        
        main_layout.addLayout(left_layout, 3)
        
        # Right side - Chat interface
        right_layout = QVBoxLayout()
        
        chat_group = QGroupBox("AI Chatbot Conversation")
        chat_layout = QVBoxLayout()
        
        # Scroll area for messages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: white; }")
        
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.addStretch()
        self.messages_layout.setSpacing(10)
        
        scroll.setWidget(self.messages_widget)
        chat_layout.addWidget(scroll)
        
        # Status indicator
        self.status_label = QLabel("🟢 Ready - Sign to chat!")
        self.status_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; font-weight: bold;")
        chat_layout.addWidget(self.status_label)
        
        chat_group.setLayout(chat_layout)
        right_layout.addWidget(chat_group)
        
        # Instructions
        instructions_group = QGroupBox("Quick Guide")
        instructions_layout = QVBoxLayout()
        
        # Enhanced mode adds more info
        if ENHANCED_MODE:
            instructions_text = (
                "✋ Basic Signs: A, B, L, O, V\n"
                "🤚 Open palm down = SPACE\n"
                "👍 Thumbs up = Submit\n"
                "⏱️  Auto-submit after 3s pause\n"
                "🔍 Bot suggests web search when needed\n"
                "🚀 Enhanced: Sentence-level recognition & ASL grammar"
            )
        else:
            instructions_text = (
                "✋ Basic Signs: A, B, L, O, V\n"
                "🤚 Open palm down = SPACE\n"
                "👍 Thumbs up = Submit\n"
                "⏱️  Auto-submit after 3s pause\n"
                "🔍 Bot suggests web search when needed"
            )
        
        instructions = QLabel(instructions_text)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 11px; color: #555;")
        instructions_layout.addWidget(instructions)
        
        instructions_group.setLayout(instructions_layout)
        instructions_group.setMaximumHeight(200)
        right_layout.addWidget(instructions_group)
        
        main_layout.addLayout(right_layout, 2)
        
        # Welcome message with mode indicator
        welcome_msg = "👋 Hello! I'm your ASL chatbot. Start signing and I'll respond to your messages!"
        if ENHANCED_MODE:
            welcome_msg += "\n\n✨ Running in Enhanced Mode with sentence-level recognition and ASL grammar support!"
        self.add_bot_message(welcome_msg)
    
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
            
        camera_index = self.camera_selector.currentIndex()
        self.camera_thread = CameraThread(camera_index, config=self.config)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.sentence_complete.connect(self.process_user_input)
        self.camera_thread.start()
        
        self.camera_selector.setEnabled(False)
        self.status_label.setText("🟢 Ready - Sign to chat!")
        self.status_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; font-weight: bold;")
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
        
        self.camera_selector.setEnabled(True)
        self.video_label.clear()
        self.video_label.setText("Camera Stopped")
        self.status_label.setText("🔴 Camera Stopped")
        self.status_label.setStyleSheet("padding: 8px; background-color: #ffebee; border-radius: 5px; font-weight: bold;")
    
    def update_frame(self, frame, gesture, word):
        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale and display
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update recognition labels
        self.gesture_label.setText(f"Gesture: {gesture}")
        self.word_label.setText(f"Signing: {word}")
    
    def process_user_input(self, text):
        """Process completed sign language input"""
        if not text.strip():
            return
        
        self.current_query = text.strip()
        
        # Add user message
        self.add_user_message(self.current_query)
        
        # Show processing status
        self.status_label.setText("🤔 Processing...")
        self.status_label.setStyleSheet("padding: 8px; background-color: #fff3e0; border-radius: 5px; font-weight: bold;")
        QApplication.processEvents()
        
        # Analyze if web search is needed
        needs_search = self.llm_handler.analyze_query_intent(self.current_query)
        
        if needs_search:
            # Bot suggests web search
            response = "I can answer that! Would you like me to search the web for the most current information?"
            self.add_bot_message(response)
            self.add_search_button(self.current_query)
        else:
            # Direct response with context
            response = self.llm_handler.respond_to_query(self.current_query, context=True)
            self.add_bot_message(response)
        
        # Reset status
        self.status_label.setText("🟢 Ready - Sign to chat!")
        self.status_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; font-weight: bold;")
    
    def perform_web_search(self, query):
        """Perform web search and respond"""
        self.status_label.setText("🔍 Searching the web...")
        self.status_label.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 5px; font-weight: bold;")
        QApplication.processEvents()
        
        # Search the web
        search_results = self.llm_handler.web_search(query)
        
        # Generate response with search results
        response = self.llm_handler.search_and_respond(query, search_results)
        
        self.add_bot_message(f"🔍 Search results:\n\n{response}")
        
        self.status_label.setText("🟢 Ready - Sign to chat!")
        self.status_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; font-weight: bold;")
    
    def add_user_message(self, text):
        """Add user message to chat"""
        msg = MessageWidget(text, is_user=True)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, msg)
        self.scroll_to_bottom()
    
    def add_bot_message(self, text):
        """Add bot message to chat"""
        msg = MessageWidget(text, is_user=False)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, msg)
        self.scroll_to_bottom()
    
    def add_search_button(self, query):
        """Add search button to chat"""
        btn_widget = SearchButtonWidget(query)
        btn_widget.search_clicked.connect(self.perform_web_search)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, btn_widget)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        QTimer.singleShot(100, lambda: self.messages_widget.parent().parent().verticalScrollBar().setValue(
            self.messages_widget.parent().parent().verticalScrollBar().maximum()
        ))
    
    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

def check_dependencies():
    """Check if all dependencies are properly installed"""
    errors = []
    
    # Check MediaPipe Tasks API availability
    try:
        # Try importing the hand landmarker task we use
        from mediapipe.tasks.python.vision import HandLandmarker  # type: ignore
    except Exception as e:
        errors.append(
            "MediaPipe Tasks API not available or misinstalled.\n"
            "If you intended to use the legacy 'mp.solutions' API, reinstall mediapipe==0.10.9,\n"
            "or install a mediapipe build that includes the Tasks API.\n"
            f"Error: {e}"
        )
    
    # Check OpenCV
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            errors.append("Cannot access camera. Make sure camera is connected and not in use.")
        cap.release()
    except Exception as e:
        errors.append(f"Camera error: {e}")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            errors.append(
                "Cannot connect to Ollama!\n"
                "Make sure Ollama is running: ollama serve"
            )
    except requests.exceptions.ConnectionError:
        errors.append(
            "Ollama not running!\n"
            "Start it with: ollama serve\n"
            "Then: ollama pull llama3.2:1b"
        )
    except Exception as e:
        errors.append(f"Ollama check error: {e}")
    
    if errors:
        print("\n" + "="*60)
        print("⚠️  STARTUP ERRORS DETECTED:")
        print("="*60)
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. {error}")
        print("\n" + "="*60)
        print("\nFix these issues and try again.\n")
        return False
    
    print("✅ All dependencies OK!")
    return True

def main():
    # Check dependencies before starting GUI
    if not check_dependencies():
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ASLChatbotApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()