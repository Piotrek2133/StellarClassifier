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

# -----------------------------
# GUI setup
# -----------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class GalaxyProducerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌌 Galaxy Classifier ✨")
        self.root.geometry("500x620")
        self.root.resizable(False, False)

        # Header
        self.header = ctk.CTkLabel(
            root, text="🌌 Galaxy Classifier ✨", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.header.pack(pady=15)

        # Features (internal name, display label with emoji)
        self.features = [
            ("u", "u"),
            ("g", "g"),
            ("r", "r"),
            ("i", "i"),
            ("z", "z")
        ]
        self.features2 = [
            ("u", "u"),
            ("g", "g"),
            ("r", "r"),
            ("i", "i"),
            ("z", "z"),
            ("redshift", "Redshift"),
            ("plate", "Plate"),
            ("MJD", "MJD")
        ]
        # Model selector (m1 or m2)
        self.model_var = ctk.StringVar(value="m1")
        model_frame = ctk.CTkFrame(root, fg_color="#2F3136", corner_radius=10)
        model_frame.pack(pady=5, padx=20, fill="x")
        model_label = ctk.CTkLabel(model_frame, text="Model:", width=100, anchor="w")
        model_label.pack(side="left", padx=5)
        self.model_menu = ctk.CTkOptionMenu(model_frame, values=["m1", "m2"], command=self.change_model)
        self.model_menu.set("m1")
        self.model_menu.pack(side="left", padx=5)

        # Container for feature input fields
        self.form_frame = ctk.CTkFrame(root, fg_color="transparent")
        self.form_frame.pack(pady=5, padx=20, fill="x")

        # Build initial fields for m1
        self.entries = {}
        self.build_fields(self.features)

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
            model = self.model_menu.get() if hasattr(self, "model_menu") else "m1"
            if model == "m1":
                feature_list = self.features
            else:
                feature_list = self.features2

            data = {field_name: float(self.entries[field_name].get())
                    for field_name, _ in feature_list}
            # Send model name as first parameter
            payload = [model, data]
            producer.send(DATA_TOPIC, value=payload)
            producer.flush()
            self.update_prediction_field("Waiting for prediction...")
        except ValueError:
            self.update_prediction_field("❌ Invalid input! Use numbers only.")

    def build_fields(self, feature_list):
        # Clear previous widgets
        for child in self.form_frame.winfo_children():
            child.destroy()
        self.entries = {}
        for field_name, label_text in feature_list:
            frame = ctk.CTkFrame(self.form_frame, fg_color="#2F3136", corner_radius=10)
            frame.pack(pady=5, fill="x")
            label = ctk.CTkLabel(frame, text=f"{label_text}:", width=100, anchor="w")
            label.pack(side="left", padx=5)
            entry = ctk.CTkEntry(frame)
            entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
            self.entries[field_name] = entry

    def change_model(self, choice):
        # Update input fields based on selected model
        if choice == "m1":
            self.build_fields(self.features)
        else:
            self.build_fields(self.features2)

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
            # class_name = CLASS_MAP.get(prediction, "Unknown")
            class_name = prediction
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
