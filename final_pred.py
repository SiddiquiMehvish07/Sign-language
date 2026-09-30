# Importing Libraries
import numpy as np
import math
import cv2

import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
from datetime import datetime
import threading

# --- Gemini AI Integration ---
import google.generativeai as genai

ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application:
class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')

        # Initialize speech engine with error handling
        try:
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("rate", 150)
            voices = self.speak_engine.getProperty("voices")
            if voices:
                self.speak_engine.setProperty("voice", voices[0].id)
            print("✓ Speech engine initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Speech engine initialization failed: {e}")
            self.speak_engine = None

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        # --- Enhanced Gemini AI Configuration ---
        self.chat_model = None
        self.chat_session = None
        self.is_ai_thinking = False

        try:
            # !!! IMPORTANT: Replace with your actual Gemini API Key !!!
            # Get your key from: https://makersuite.google.com/app/apikey
            api_key = "AIzaSyAkarQyCHX2C34OLlxtM4tuPv8jZuOYN9k"  # ← PUT YOUR KEY HERE

            if api_key == "YOUR_API_KEY_HERE" or api_key == "":
                print("⚠ WARNING: Using placeholder API key. Please replace with your actual Gemini API key!")
                print("   Get your key from: https://makersuite.google.com/app/apikey")
                raise Exception("No valid API key provided")

            genai.configure(api_key=api_key)

            # Use the latest model with optimized settings
            generation_config = {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            self.chat_model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config=generation_config
            )

            # Start a chat session for continuous conversation
            self.chat_session = self.chat_model.start_chat(history=[])
            print("✓ Gemini AI configured successfully")
        except Exception as e:
            print(f"✗ Gemini API configuration failed. Chatbot disabled. Error: {e}")
            self.chat_model = None
            self.chat_session = None

        for i in ascii_uppercase:
            self.ct[i] = 0
        print("✓ Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language AI Assistant - Powered by Gemini")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x1050")  # Increased height
        self.root.configure(bg='#0a0e27')

        # Add custom fonts
        self.title_font = ("Segoe UI", 22, "bold")
        self.header_font = ("Segoe UI", 14, "bold")
        self.body_font = ("Segoe UI", 11)
        self.small_font = ("Segoe UI", 9)

        # === LEFT PANEL: VIDEO FEEDS ===
        left_frame = tk.Frame(self.root, bg='#0a0e27')
        left_frame.place(x=20, y=20, width=520, height=740)

        # Live Video Panel with gradient effect
        video_panel = tk.Frame(left_frame, bg='#1a1f3a', relief=tk.FLAT)
        video_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        video_header = tk.Frame(video_panel, bg='#1a1f3a', height=60)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)

        header_content = tk.Frame(video_header, bg='#1a1f3a')
        header_content.pack(expand=True)

        tk.Label(header_content, text="📹", font=("Segoe UI", 20),
                 bg='#1a1f3a').pack(side=tk.LEFT, padx=(10, 8))

        tk.Label(header_content, text="Live Camera Feed",
                 font=self.header_font, fg='#00d9ff', bg='#1a1f3a').pack(side=tk.LEFT)

        video_container = tk.Frame(video_panel, bg='#0f1729', relief=tk.FLAT, bd=2)
        video_container.pack(padx=15, pady=(0, 15), fill=tk.BOTH, expand=True)

        self.panel = tk.Label(video_container, bg='#000000', relief=tk.FLAT)
        self.panel.pack(padx=3, pady=3, fill=tk.BOTH, expand=True)

        # Skeleton Panel
        skeleton_panel = tk.Frame(left_frame, bg='#1a1f3a', relief=tk.FLAT)
        skeleton_panel.pack(fill=tk.X)

        skeleton_header = tk.Frame(skeleton_panel, bg='#1a1f3a', height=50)
        skeleton_header.pack(fill=tk.X)
        skeleton_header.pack_propagate(False)

        skel_header_content = tk.Frame(skeleton_header, bg='#1a1f3a')
        skel_header_content.pack(expand=True)

        tk.Label(skel_header_content, text="🖐️", font=("Segoe UI", 18),
                 bg='#1a1f3a').pack(side=tk.LEFT, padx=(10, 8))

        tk.Label(skel_header_content, text="Hand Skeleton Detection",
                 font=self.header_font, fg='#00d9ff', bg='#1a1f3a').pack(side=tk.LEFT)

        skeleton_container = tk.Frame(skeleton_panel, bg='#0f1729', relief=tk.FLAT, bd=2)
        skeleton_container.pack(padx=15, pady=(0, 15))

        self.panel2 = tk.Label(skeleton_container, bg='#000000', relief=tk.FLAT)
        self.panel2.pack(padx=3, pady=3)
        self.panel2.config(width=320, height=320)

        # === MIDDLE PANEL: TRANSLATION & CONTROLS ===
        middle_frame = tk.Frame(self.root, bg='#1a1f3a', relief=tk.FLAT, bd=0)
        middle_frame.place(x=560, y=20, width=820, height=900)  # Increased height

        # Header with modern styling
        header_panel = tk.Frame(middle_frame, bg='#0f1729', height=90)
        header_panel.pack(fill=tk.X, pady=(0, 15))
        header_panel.pack_propagate(False)

        title_container = tk.Frame(header_panel, bg='#0f1729')
        title_container.pack(expand=True)

        tk.Label(title_container, text="✨ Sign Language Translation",
                 font=self.title_font, fg='#00d9ff', bg='#0f1729').pack()

        tk.Label(title_container, text="Real-time ASL to Text Conversion",
                 font=self.small_font, fg='#808080', bg='#0f1729').pack(pady=(3, 0))

        # Current Character Display with animation effect
        char_panel = tk.Frame(middle_frame, bg='#0f1729', relief=tk.FLAT, bd=2)
        char_panel.pack(pady=12, padx=25, fill=tk.X)

        char_content = tk.Frame(char_panel, bg='#0f1729')
        char_content.pack(pady=15, padx=25)

        tk.Label(char_content, text="Current Sign:", font=("Segoe UI", 13, "bold"),
                 fg='#a0a0a0', bg='#0f1729').pack(side=tk.LEFT, padx=12)

        char_display = tk.Frame(char_content, bg='#162236', relief=tk.FLAT, bd=2)
        char_display.pack(side=tk.LEFT, padx=15)

        self.panel3 = tk.Label(char_display, text="—", font=("Segoe UI", 32, "bold"),
                               fg='#00ff88', bg='#162236', width=4,
                               padx=20, pady=8)
        self.panel3.pack()

        # Sentence Display with modern scrollbar
        sentence_panel = tk.Frame(middle_frame, bg='#0f1729', relief=tk.FLAT, bd=2)
        sentence_panel.pack(pady=12, fill=tk.X, padx=25)

        sentence_header = tk.Frame(sentence_panel, bg='#0f1729')
        sentence_header.pack(fill=tk.X, pady=(15, 8), padx=20)

        tk.Label(sentence_header, text="📝 Translated Sentence",
                 font=("Segoe UI", 13, "bold"), fg='#ffffff', bg='#0f1729').pack(anchor='w')

        text_container = tk.Frame(sentence_panel, bg='#162236', relief=tk.FLAT, bd=1)
        text_container.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.panel5 = tk.Text(text_container, font=("Segoe UI", 16),
                              fg='#ffffff', bg='#162236', wrap=tk.WORD,
                              height=4, relief=tk.FLAT, bd=0, padx=15, pady=15,
                              insertbackground='#00d9ff')
        self.panel5.pack(fill=tk.X)

        # Action Buttons with hover effects
        button_panel = tk.Frame(middle_frame, bg='#1a1f3a')
        button_panel.pack(pady=15)

        button_configs = [
            ("🔊 Speak", '#4CAF50', '#45a049', self.speak_fun),
            ("⎵ Space", '#FF9800', '#e68900', self.add_space),
            ("🗑️ Clear", '#f44336', '#da190b', self.clear_fun),
            ("🤖 Ask AI", '#9C27B0', '#7B1FA2', self.auto_ask_ai),  # Purple for visibility
        ]

        for idx, (text, bg, active_bg, cmd) in enumerate(button_configs):
            btn = tk.Button(button_panel, text=text,
                            font=("Segoe UI", 11, "bold"),
                            bg=bg, fg='white',
                            activebackground=active_bg,
                            width=12, height=2,
                            relief=tk.FLAT, bd=0,
                            cursor='hand2',
                            command=cmd)
            btn.grid(row=0, column=idx, padx=6, pady=5)

        # AI Response Display - MOVED BEFORE SUGGESTIONS
        ai_response_panel = tk.Frame(middle_frame, bg='#0f1729', relief=tk.FLAT, bd=2)
        ai_response_panel.pack(pady=15, fill=tk.BOTH, expand=True, padx=25)

        ai_response_header = tk.Frame(ai_response_panel, bg='#0f1729')
        ai_response_header.pack(fill=tk.X, pady=(15, 8), padx=20)

        tk.Label(ai_response_header, text="🤖 AI Assistant",
                 font=("Segoe UI", 13, "bold"), fg='#00d9ff', bg='#0f1729').pack(anchor='w')

        ai_response_container = tk.Frame(ai_response_panel, bg='#162236', relief=tk.FLAT, bd=2)
        ai_response_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.ai_response_display = scrolledtext.ScrolledText(
            ai_response_container,
            font=("Segoe UI", 11),
            fg='#ffffff', bg='#162236',
            wrap=tk.WORD,
            height=10,  # Fixed height for better visibility
            relief=tk.FLAT, bd=0,
            padx=15, pady=15,
            state=tk.DISABLED
        )
        self.ai_response_display.pack(fill=tk.BOTH, expand=True)

        # Word Suggestions with modern cards
        suggest_panel = tk.Frame(middle_frame, bg='#0f1729', relief=tk.FLAT, bd=2)
        suggest_panel.pack(pady=12, fill=tk.X, padx=25)

        suggest_header = tk.Frame(suggest_panel, bg='#0f1729')
        suggest_header.pack(fill=tk.X, pady=(15, 10), padx=20)

        tk.Label(suggest_header, text="💡 Word Suggestions",
                 font=("Segoe UI", 12, "bold"), fg='#ffaa00', bg='#0f1729').pack(anchor='w')

        suggestion_grid = tk.Frame(suggest_panel, bg='#0f1729')
        suggestion_grid.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.suggestion_buttons = []
        for i in range(4):
            btn = tk.Button(suggestion_grid,
                            font=("Segoe UI", 10),
                            bg='#162236', fg='#ffffff',
                            activebackground='#1e3a52',
                            relief=tk.FLAT, bd=0,
                            cursor='hand2', pady=10,
                            command=[self.action1, self.action2, self.action3, self.action4][i])
            btn.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky='ew')
            self.suggestion_buttons.append(btn)

        suggestion_grid.columnconfigure(0, weight=1)
        suggestion_grid.columnconfigure(1, weight=1)

        self.b1, self.b2, self.b3, self.b4 = self.suggestion_buttons

        # === BOTTOM STATUS BAR ===
        status_bar = tk.Frame(self.root, bg='#0f1729', relief=tk.FLAT, height=60)
        status_bar.place(x=20, y=970, width=1360, height=60)  # Moved down

        status_content = tk.Frame(status_bar, bg='#0f1729')
        status_content.pack(expand=True)

        self.status_indicator = tk.Label(status_content, text="●",
                                         font=("Arial", 18),
                                         fg='#00ff88', bg='#0f1729')
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 12))

        status_text_frame = tk.Frame(status_content, bg='#0f1729')
        status_text_frame.pack(side=tk.LEFT)

        self.status_label = tk.Label(status_text_frame, text="Ready - Start signing!",
                                     font=("Segoe UI", 12, "bold"),
                                     fg='#ffffff', bg='#0f1729')
        self.status_label.pack(anchor='w')

        self.status_detail = tk.Label(status_text_frame,
                                      text="Camera active • AI " + ("ready" if self.chat_session else "unavailable"),
                                      font=self.small_font,
                                      fg='#808080', bg='#0f1729')
        self.status_detail.pack(anchor='w')

        # Initialize variables
        self.str = ""
        self.ccc = 0
        self.word = ""
        self.current_symbol = ""
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""

        # Initialize AI response area
        self.display_ai_welcome()

        self.video_loop()

    def display_ai_welcome(self):
        """Display welcome message in AI response area"""
        if self.chat_session:
            welcome_msg = ("👋 Welcome to AI Assistant!\n\n"
                           "How to use:\n"
                           "1. Sign your question using hand gestures\n"
                           "2. Watch it appear in 'Translated Sentence'\n"
                           "3. Click '🤖 Ask AI' button\n"
                           "4. Get instant AI responses here!\n\n"
                           "Example: Sign 'WHAT IS PYTHON' or 'HELLO WORLD'")
        else:
            welcome_msg = ("❌ AI Assistant Unavailable\n\n"
                           "Please add your Gemini API key to enable AI features.\n\n"
                           "Steps:\n"
                           "1. Visit: https://makersuite.google.com/app/apikey\n"
                           "2. Get your free API key\n"
                           "3. Replace 'YOUR_API_KEY_HERE' in code (line ~73)\n"
                           "4. Restart the application")

        self.ai_response_display.config(state=tk.NORMAL)
        self.ai_response_display.delete('1.0', tk.END)
        self.ai_response_display.insert('1.0', welcome_msg)
        self.ai_response_display.config(state=tk.DISABLED)

    def auto_ask_ai(self):
        """Automatically send the signed sentence to AI and display response"""
        sentence = self.str.strip()

        if not sentence:
            self.update_status("⚠️ No text to send", "warning", "Sign something first!")
            self.ai_response_display.config(state=tk.NORMAL)
            self.ai_response_display.delete('1.0', tk.END)
            self.ai_response_display.insert('1.0',
                                            "⚠️ Please sign a question first!\n\n"
                                            "Try signing:\n"
                                            "• HELLO\n"
                                            "• WHAT IS AI\n"
                                            "• HELP ME\n\n"
                                            "Then click '🤖 Ask AI' button")
            self.ai_response_display.config(state=tk.DISABLED)
            return

        if not self.chat_session:
            self.ai_response_display.config(state=tk.NORMAL)
            self.ai_response_display.delete('1.0', tk.END)
            self.ai_response_display.insert('1.0',
                                            "❌ AI is unavailable!\n\n"
                                            "Possible reasons:\n"
                                            "1. No API key configured\n"
                                            "2. Invalid API key\n"
                                            "3. No internet connection\n"
                                            "4. API quota exceeded\n\n"
                                            "Get your free API key:\n"
                                            "https://makersuite.google.com/app/apikey\n\n"
                                            "Then replace 'YOUR_API_KEY_HERE' in code")
            self.ai_response_display.config(state=tk.DISABLED)
            self.update_status("AI unavailable", "error", "Check API key")
            return

        # Show thinking status
        self.ai_response_display.config(state=tk.NORMAL)
        self.ai_response_display.delete('1.0', tk.END)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ai_response_display.insert('1.0',
                                        f"[{timestamp}] 📝 Your Question:\n{sentence}\n\n"
                                        f"🤔 AI is thinking...\n"
                                        f"⏳ Processing your request...")
        self.ai_response_display.config(state=tk.DISABLED)
        self.update_status("Processing...", "thinking", f"Analyzing: {sentence[:30]}...")

        # Process in thread
        def get_ai_answer():
            try:
                response = self.chat_session.send_message(sentence)
                answer = response.text
                self.root.after(0, lambda: self.show_ai_response(sentence, answer))
            except Exception as e:
                error_msg = f"{str(e)}"
                self.root.after(0, lambda: self.show_ai_error(sentence, error_msg))

        threading.Thread(target=get_ai_answer, daemon=True).start()

    def show_ai_response(self, question, answer):
        """Display AI response in the main window"""
        self.ai_response_display.config(state=tk.NORMAL)
        self.ai_response_display.delete('1.0', tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        display_text = (f"[{timestamp}] 📝 Your Question:\n{question}\n\n"
                        f"🤖 AI Answer:\n{answer}\n\n"
                        f"{'─' * 50}\n")

        self.ai_response_display.insert('1.0', display_text)
        self.ai_response_display.see('1.0')  # Scroll to top
        self.ai_response_display.config(state=tk.DISABLED)

        self.update_status("✓ AI response received!", "success", "Answer displayed above")

    def show_ai_error(self, question, error):
        """Display error in AI response area"""
        self.ai_response_display.config(state=tk.NORMAL)
        self.ai_response_display.delete('1.0', tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        display_text = (f"[{timestamp}] 📝 Your Question:\n{question}\n\n"
                        f"❌ Error occurred:\n{error}\n\n"
                        f"Please try again or check:\n"
                        f"• Internet connection\n"
                        f"• API key validity\n"
                        f"• API quota limits")

        self.ai_response_display.insert('1.0', display_text)
        self.ai_response_display.config(state=tk.DISABLED)

        self.update_status("AI request failed", "error", "Please try again")

    def update_status(self, message, status_type="info", detail=""):
        """Update the status bar with color coding"""
        colors = {
            "success": "#00ff88",
            "error": "#ff4444",
            "warning": "#ffaa00",
            "info": "#00d9ff",
            "thinking": "#ff9800"
        }
        self.status_label.config(text=message)
        self.status_indicator.config(fg=colors.get(status_type, "#ffffff"))
        if detail:
            self.status_detail.config(text=detail)

    def add_space(self):
        """Add a space to the sentence"""
        self.str = self.str + " "
        self.update_status("Space added", "success")

    def video_loop(self):
        ok, frame = self.vs.read()

        if not ok:
            print("Error reading frame from video stream.")
            self.root.after(1, self.video_loop)
            return

        try:
            cv2image = cv2.flip(frame, 1)
            cv2image_copy = np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            hands = hd.findHands(cv2image_copy, draw=False, flipType=False)

            if hands:
                hand = hands[0]
                map_data = hand[0] if isinstance(hand, list) else hand
                x, y, w, h = map_data['bbox']

                y_start = max(0, y - offset)
                y_end = min(cv2image_copy.shape[0], y + h + offset)
                x_start = max(0, x - offset)
                x_end = min(cv2image_copy.shape[1], x + w + offset)

                image = cv2image_copy[y_start:y_end, x_start:x_end]
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255

                if image.size > 0:
                    handz = hd2.findHands(image, draw=False, flipType=True)
                    self.ccc += 1

                    if handz:
                        hand = handz[0]
                        handmap = hand[0] if isinstance(hand, list) else hand
                        self.pts = handmap['lmList']

                        os_val = ((400 - image.shape[1]) // 2)
                        os1 = ((400 - image.shape[0]) // 2)

                        # Drawing skeleton
                        for t in range(0, 4, 1):
                            cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1),
                                     (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(5, 8, 1):
                            cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1),
                                     (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(9, 12, 1):
                            cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1),
                                     (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(13, 16, 1):
                            cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1),
                                     (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1), (0, 255, 0), 3)
                        for t in range(17, 20, 1):
                            cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1),
                                     (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1), (0, 255, 0), 3)

                        cv2.line(white, (self.pts[5][0] + os_val, self.pts[5][1] + os1),
                                 (self.pts[9][0] + os_val, self.pts[9][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[9][0] + os_val, self.pts[9][1] + os1),
                                 (self.pts[13][0] + os_val, self.pts[13][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[13][0] + os_val, self.pts[13][1] + os1),
                                 (self.pts[17][0] + os_val, self.pts[17][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_val, self.pts[0][1] + os1),
                                 (self.pts[5][0] + os_val, self.pts[5][1] + os1), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_val, self.pts[0][1] + os1),
                                 (self.pts[17][0] + os_val, self.pts[17][1] + os1), (0, 255, 0), 3)

                        for i in range(21):
                            cv2.circle(white, (self.pts[i][0] + os_val, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                        res = white
                        self.predict(res)

                        self.current_image2 = Image.fromarray(res)
                        imgtk = ImageTk.PhotoImage(image=self.current_image2)
                        self.panel2.imgtk = imgtk
                        self.panel2.config(image=imgtk)

                        self.panel3.config(text=self.current_symbol)

                        self.b1.config(text=self.word1 if self.word1 else "—")
                        self.b2.config(text=self.word2 if self.word2 else "—")
                        self.b3.config(text=self.word3 if self.word3 else "—")
                        self.b4.config(text=self.word4 if self.word4 else "—")

            # Update sentence display
            self.panel5.delete('1.0', tk.END)
            self.panel5.insert('1.0', self.str)

        except Exception as e:
            print(f"An error occurred in video_loop: {e}")

        finally:
            self.root.after(1, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        if not self.word1:
            return
        idx_word = self.str.rfind(self.word.strip())
        if idx_word == -1:
            last_space_index = self.str.rfind(" ")
            idx_word = last_space_index + 1 if last_space_index != -1 else 0
        if idx_word > 0 and self.str[idx_word - 1] == ' ':
            idx_word -= 1
        self.str = self.str[:idx_word] + self.word1.upper()
        self.word = self.word1.upper()
        self.update_status(f"Selected: {self.word1}", "success")

    def action2(self):
        if not self.word2:
            return
        idx_word = self.str.rfind(self.word.strip())
        if idx_word == -1:
            last_space_index = self.str.rfind(" ")
            idx_word = last_space_index + 1 if last_space_index != -1 else 0
        if idx_word > 0 and self.str[idx_word - 1] == ' ':
            idx_word -= 1
        self.str = self.str[:idx_word] + self.word2.upper()
        self.word = self.word2.upper()
        self.update_status(f"Selected: {self.word2}", "success")

    def action3(self):
        if not self.word3:
            return
        idx_word = self.str.rfind(self.word.strip())
        if idx_word == -1:
            last_space_index = self.str.rfind(" ")
            idx_word = last_space_index + 1 if last_space_index != -1 else 0
        if idx_word > 0 and self.str[idx_word - 1] == ' ':
            idx_word -= 1
        self.str = self.str[:idx_word] + self.word3.upper()
        self.word = self.word3.upper()
        self.update_status(f"Selected: {self.word3}", "success")

    def action4(self):
        if not self.word4:
            return
        idx_word = self.str.rfind(self.word.strip())
        if idx_word == -1:
            last_space_index = self.str.rfind(" ")
            idx_word = last_space_index + 1 if last_space_index != -1 else 0
        if idx_word > 0 and self.str[idx_word - 1] == ' ':
            idx_word -= 1
        self.str = self.str[:idx_word] + self.word4.upper()
        self.word = self.word4.upper()
        self.update_status(f"Selected: {self.word4}", "success")

    def speak_fun(self):
        """Text to speech function with thread safety"""
        text = self.str.strip()
        if not text:
            self.update_status("No text to speak", "warning")
            return

        if not self.speak_engine:
            self.update_status("Speech engine not available", "error")
            return

        try:
            self.update_status("Speaking...", "info", "Text-to-speech active")

            def speak_thread():
                try:
                    self.speak_engine.say(text)
                    self.speak_engine.runAndWait()
                    self.root.after(100, lambda: self.update_status(
                        "Speaking completed", "success", "Ready for more input"))
                except Exception as e:
                    print(f"Speech error: {e}")
                    self.root.after(100, lambda: self.update_status(
                        "Speech failed", "error", "Please check audio settings"))

            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()

        except Exception as e:
            print(f"Speech initialization error: {e}")
            self.update_status("Speech engine error", "error")

    def clear_fun(self):
        """Clear the sentence and suggestions"""
        self.str = ""
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""
        self.update_status("Sentence cleared", "info", "Ready for new input")

    def predict(self, test_image):
        """Predict sign language gesture from hand skeleton"""
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # All prediction logic from original code
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][
                0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > \
                    self.pts[4][0]:
                ch1 = 2

        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < \
                    self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4

        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > \
                    self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][
                0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] >
                    self.pts[20][1]):
                ch1 = 1

        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and
                    self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and
                    self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and ((
                    self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                 self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][
                0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] >
                    self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4],
                                                                                            self.pts[11]) < 50:
                ch1 = 1

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][
                1]:
                ch1 = 1

        # Subgroup classifications
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][
                0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][
                0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < \
                    self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][
                1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][
                0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][
                1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][
                0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        if ch1 in (1, 'E', 'S', 'X', 'Y', 'B'):
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] <
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        if ch1 in ('E', 'Y', 'B'):
            if (self.pts[4][0] < self.pts[5][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] >
                    self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        if ch1 in ('next', 'B', 'C', 'H', 'F', 'X'):
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and (
                    self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] <
                    self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (
                    self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] <
                    self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count - 2) % 10] != "next":
                if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                    self.str = self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

        # Word suggestions with enchant dictionary
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st + 1:ed]
            self.word = word
            if len(word.strip()) != 0:
                try:
                    ddd.check(word)
                    suggestions = ddd.suggest(word)
                    lenn = len(suggestions)
                    self.word1 = self.word2 = self.word3 = self.word4 = ""

                    if lenn >= 4:
                        self.word4 = suggestions[3]
                    if lenn >= 3:
                        self.word3 = suggestions[2]
                    if lenn >= 2:
                        self.word2 = suggestions[1]
                    if lenn >= 1:
                        self.word1 = suggestions[0]
                except Exception as e:
                    print(f"Dictionary suggestion error: {e}")
                    self.word1 = self.word2 = self.word3 = self.word4 = ""
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = ""
        else:
            self.word1 = self.word2 = self.word3 = self.word4 = ""

    def destructor(self):
        """Clean up resources on application close"""
        print("Shutting down application...")
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Sign Language AI Assistant...")
    print("=" * 60)
    print("\nInitializing components:")
    print("- Loading camera feed")
    print("- Loading hand detection model")
    print("- Initializing AI chatbot")
    print("\nPlease wait...\n")

    try:
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"\nError starting application: {e}")
        traceback.print_exc()