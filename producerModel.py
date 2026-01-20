import customtkinter as ctk
from kafka import KafkaProducer, KafkaConsumer
import json
import threading
import queue

# -----------------------------
# Kafka configuration
# -----------------------------
DATA_TOPIC = "galaxy-input"
PRED_TOPIC = "galaxy-prediction"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

consumer = KafkaConsumer(
    PRED_TOPIC,
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    group_id="producer-gui-modern-group",
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

# Thread-safe queue for predictions
pred_queue = queue.Queue()

# Map numeric predictions to human-readable classes
CLASS_MAP = {
    0: "Galaxy",
    1: "Star",
    2: "Quasar"
}

# -----------------------------
# GUI setup
# -----------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class GalaxyProducerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌌 Galaxy Classifier ✨")
        self.root.geometry("500x580")
        self.root.resizable(False, False)

        # Header
        self.header = ctk.CTkLabel(
            root, text="🌌 Galaxy Classifier ✨", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.header.pack(pady=15)

        # Features (internal name, display label with emoji)
        self.features = [
            ("u", "💜 Ultraviolet"),
            ("g", "💚 Green"),
            ("r", "❤️ Red"),
            ("i", "🔴 Near Infrared"),
            ("z", "🟥 Infrared"),
            ("redshift", "🌈 Redshift"),
            ("plate", "🛡️ Plate ID"),
            ("MJD", "📅 MJD")
        ]
        self.entries = {}
        for field_name, label_text in self.features:
            frame = ctk.CTkFrame(root, fg_color="#2F3136", corner_radius=10)
            frame.pack(pady=5, padx=20, fill="x")
            label = ctk.CTkLabel(
                frame, text=f"{label_text}:", width=100, anchor="w")
            label.pack(side="left", padx=5)
            entry = ctk.CTkEntry(frame)
            entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
            self.entries[field_name] = entry

        # Send button
        self.send_button = ctk.CTkButton(
            root,
            text="🚀 Send & Predict",
            command=self.send_data,
            fg_color="#7289DA",
            hover_color="#5b6eae",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.send_button.pack(pady=15)

        # Prediction output field (centered, read-only)
        self.pred_field = ctk.CTkEntry(
            root,
            placeholder_text="Prediction will appear here",
            state="disabled",
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        self.pred_field.pack(pady=10, padx=20, fill="x")

        # Start thread to listen for predictions
        threading.Thread(target=self.listen_predictions, daemon=True).start()
        # Start periodic check of the queue
        self.root.after(100, self.check_queue)

    # -----------------------------
    # Send input data to Kafka
    # -----------------------------
    def send_data(self):
        try:
            data = {field_name: float(self.entries[field_name].get())
                    for field_name, _ in self.features}
            producer.send(DATA_TOPIC, value=data)
            producer.flush()
            self.update_prediction_field("Waiting for prediction...")
        except ValueError:
            self.update_prediction_field("❌ Invalid input! Use numbers only.")

    # -----------------------------
    # Kafka listener for predictions
    # -----------------------------
    def listen_predictions(self):
        for message in consumer:
            pred_queue.put(message.value)

    # -----------------------------
    # Check queue and update GUI
    # -----------------------------
    def check_queue(self):
        while not pred_queue.empty():
            prediction = pred_queue.get()
            class_name = CLASS_MAP.get(prediction, "Unknown")
            self.update_prediction_field(class_name)
        self.root.after(100, self.check_queue)

    # -----------------------------
    # Update the prediction field
    # -----------------------------
    def update_prediction_field(self, text):
        self.pred_field.configure(state="normal")
        self.pred_field.delete(0, "end")
        self.pred_field.insert(0, text)
        self.pred_field.configure(state="disabled")


# -----------------------------
# Run the GUI
# -----------------------------
if __name__ == "__main__":
    root = ctk.CTk()
    app = GalaxyProducerApp(root)
    root.mainloop()
