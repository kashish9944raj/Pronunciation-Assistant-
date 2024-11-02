import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk
import random

# Sample trained sentences for demonstration
sample_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A picture is worth a thousand words."
]

class PronunciationDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pronunciation Detector for Kids")
        self.root.geometry("600x600")
        self.root.configure(bg="#add8e6")  # Light blue background

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.random_sentence_tab = ttk.Frame(self.notebook)
        self.custom_sentence_tab = ttk.Frame(self.notebook)
        self.record_pronunciation_tab = ttk.Frame(self.notebook)
        self.submit_answer_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.random_sentence_tab, text='Random Sentence')
        self.notebook.add(self.custom_sentence_tab, text='Custom Sentence')
        self.notebook.add(self.record_pronunciation_tab, text='Record Pronunciation')
        self.notebook.add(self.submit_answer_tab, text='Submit Answer')

        # Initialize each tab
        self.create_random_sentence_tab()
        self.create_custom_sentence_tab()
        self.create_record_pronunciation_tab()
        self.create_submit_answer_tab()

    def create_random_sentence_tab(self):
        # Random Sentence Tab
        self.title_label = tk.Label(self.random_sentence_tab, text="Get Random Sentence", font=("Comic Sans MS", 20, "bold"), bg="#add8e6", fg="#ff6347")
        self.title_label.pack(pady=20)

        self.random_sentence_btn = tk.Button(self.random_sentence_tab, text="Get Random Sentence", command=self.get_random_sentence, 
                                              font=("Comic Sans MS", 16), bg="#ff6347", fg="white", borderwidth=0,
                                              relief="groove", padx=20, pady=10)
        self.random_sentence_btn.pack(pady=20)

    def create_custom_sentence_tab(self):
        # Custom Sentence Tab
        self.title_label = tk.Label(self.custom_sentence_tab, text="Type Your Own Sentence", font=("Comic Sans MS", 20, "bold"), bg="#add8e6", fg="#ff6347")
        self.title_label.pack(pady=20)

        self.custom_sentence_input = scrolledtext.ScrolledText(self.custom_sentence_tab, width=40, height=5, bg="#ffffff", font=("Arial", 12), wrap=tk.WORD)
        self.custom_sentence_input.pack(pady=10)

        self.check_grammar_btn = tk.Button(self.custom_sentence_tab, text="Check Grammar", command=self.check_grammar, 
                                            font=("Comic Sans MS", 16), bg="#ff6347", fg="white", borderwidth=0,
                                            relief="groove", padx=20, pady=10)
        self.check_grammar_btn.pack(pady=20)

        # Parts of Speech Output
        self.pos_label = tk.Label(self.custom_sentence_tab, text="Parts of Speech: ", font=("Arial", 12), bg="#add8e6")
        self.pos_label.pack(pady=5)

        # Score Graph Placeholder
        self.score_label = tk.Label(self.custom_sentence_tab, text="Score Graph: [Placeholder]", font=("Arial", 12), bg="#add8e6")
        self.score_label.pack(pady=10)

    def create_record_pronunciation_tab(self):
        # Record Pronunciation Tab
        self.title_label = tk.Label(self.record_pronunciation_tab, text="Record Your Pronunciation", font=("Comic Sans MS", 20, "bold"), bg="#add8e6", fg="#ff6347")
        self.title_label.pack(pady=20)

        self.record_btn = tk.Button(self.record_pronunciation_tab, text="Record Pronunciation", command=self.record_pronunciation, 
                                     font=("Comic Sans MS", 16), bg="#ff6347", fg="white", borderwidth=0,
                                     relief="groove", padx=20, pady=10)
        self.record_btn.pack(pady=20)

    def create_submit_answer_tab(self):
        # Submit Answer Tab
        self.title_label = tk.Label(self.submit_answer_tab, text="Submit Your Answer", font=("Comic Sans MS", 20, "bold"), bg="#add8e6", fg="#ff6347")
        self.title_label.pack(pady=20)

        self.answer_input = scrolledtext.ScrolledText(self.submit_answer_tab, width=40, height=5, bg="#ffffff", font=("Arial", 12), wrap=tk.WORD)
        self.answer_input.pack(pady=10)

        self.submit_btn = tk.Button(self.submit_answer_tab, text="Submit Answer", command=self.submit_answer, 
                                     font=("Comic Sans MS", 16), bg="#ff6347", fg="white", borderwidth=0,
                                     relief="groove", padx=20, pady=10)
        self.submit_btn.pack(pady=20)

    def get_random_sentence(self):
        sentence = random.choice(sample_sentences)
        messagebox.showinfo("Random Sentence", sentence)

    def check_grammar(self):
        sentence = self.custom_sentence_input.get("1.0", tk.END).strip()
        # Placeholder for grammar checking logic
        if sentence:
            self.pos_label.config(text="Parts of Speech: Noun, Verb, Adjective")  # Placeholder output
            self.score_label.config(text="Score Graph: [Placeholder]")  # Placeholder for graph
        else:
            messagebox.showwarning("Warning", "Please enter a sentence.")

    def record_pronunciation(self):
        messagebox.showinfo("Record Pronunciation", "Recording started... (Placeholder)")

    def submit_answer(self):
        answer = self.answer_input.get("1.0", tk.END).strip()
        if answer:
            messagebox.showinfo("Submit Answer", f"Your answer has been submitted: {answer}")
        else:
            messagebox.showwarning("Warning", "Please enter your answer.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PronunciationDetectorApp(root)
    root.mainloop()
