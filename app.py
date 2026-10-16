
from flask import Flask, render_template_string, request
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64

from ai_edge_litert.interpreter import Interpreter

from database import (
    initialize_database,
    save_prediction,
    get_dashboard_data
)

app = Flask(__name__)

# -------------------- LOAD AI MODEL --------------------

MODEL_PATH = Path(__file__).parent / "waste_classifier.tflite"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Must match the alphabetical class order used during training.
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Create the database table if it does not exist.
initialize_database()


# -------------------- WASTE TREATMENT GUIDE --------------------

waste_guide = {
    "cardboard": {
        "method": "Composting or recycling",
        "steps": [
            "Remove tape and plastic.",
            "Tear clean cardboard into small pieces.",
            "Add suitable cardboard to a compost pile or send it for recycling."
        ],
        "decomposition": (
            "Cardboard may break down in a few weeks to several months "
            "under suitable composting conditions."
        ),
        "tip": "Keep cardboard dry when storing it for recycling."
    },

    "glass": {
        "method": "Recycling",
        "steps": [
            "Empty and rinse the glass container.",
            "Separate it according to local recycling rules.",
            "Take it to a designated glass recycling collection point."
        ],
        "decomposition": (
            "Glass does not naturally decompose within a practical "
            "human timescale. Recycling is recommended."
        ),
        "tip": "Handle broken glass carefully."
    },

    "metal": {
        "method": "Recycling",
        "steps": [
            "Empty the metal can or container.",
            "Rinse it if necessary.",
            "Place it in the appropriate metal recycling bin."
        ],
        "decomposition": (
            "Metal corrosion depends on the type of metal and its "
            "environment. Recycling helps conserve raw materials."
        ),
        "tip": "Be careful of sharp metal edges."
    },

    "paper": {
        "method": "Composting or recycling",
        "steps": [
            "Separate clean paper from other waste.",
            "Tear it into smaller pieces if composting.",
            "Add suitable paper to compost or send it for recycling."
        ],
        "decomposition": (
            "Paper may break down in a few weeks to several months, "
            "depending on its type and environmental conditions."
        ),
        "tip": "Avoid composting paper contaminated with chemicals."
    },

    "plastic": {
        "method": "Reuse or recycling",
        "steps": [
            "Check the plastic type and local recycling rules.",
            "Empty and clean the item if required.",
            "Reuse it safely or place it in the correct recycling bin."
        ],
        "decomposition": (
            "Many plastics persist for decades or longer and may "
            "fragment into microplastics rather than fully decompose."
        ),
        "tip": "Never burn plastic or add it to ordinary compost."
    },

    "trash": {
        "method": "Separate and dispose responsibly",
        "steps": [
            "Check the material of the item.",
            "Separate recyclable and compostable materials.",
            "Put remaining waste in the designated residual-waste bin.",
            "Take batteries and electronic waste to authorized collection points."
        ],
        "decomposition": (
            "Decomposition depends on the actual material. The trash "
            "category does not identify the exact material."
        ),
        "tip": "Keep hazardous waste separate from ordinary household waste."
    }
}


# -------------------- COMMON PAGE STYLES --------------------

STYLE = """
<style>
    * { box-sizing: border-box; }

    body {
        font-family: Arial, sans-serif;
        background: #eef7f0;
        margin: 0;
        padding: 24px 14px;
        color: #26382b;
    }

    .container {
        max-width: 900px;
        margin: auto;
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 18px #0001;
    }

    h1, h2, h3 { color: #217346; }

    p { color: #555; line-height: 1.6; }

    .nav {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
    }

    .button {
        display: inline-block;
        background: #217346;
        color: white;
        padding: 12px 18px;
        text-decoration: none;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 15px;
    }

    .button:hover { background: #185b36; }

    input[type=file] {
        margin: 20px 0;
        max-width: 100%;
    }

    .result, .card {
        background: #e8f5e9;
        padding: 18px;
        border-radius: 12px;
        margin: 18px 0;
    }

    .result { text-align: left; }
    .result h2 { text-align: center; }

    .uploaded-image {
        display: block;
        max-width: 100%;
        max-height: 300px;
        margin: 15px auto;
        border-radius: 10px;
    }

    .method {
        background: #d1ead6;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }

    li { margin-bottom: 10px; line-height: 1.5; }

    .tip {
        background: #fff8df;
        padding: 12px;
        border-radius: 8px;
        line-height: 1.6;
    }

    .stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 14px;
        margin: 20px 0;
    }

    .stat-card {
        background: #e8f5e9;
        padding: 20px 12px;
        border-radius: 12px;
        text-align: center;
    }

    .stat-number {
        font-size: 30px;
        font-weight: bold;
        color: #217346;
        margin: 8px 0;
    }

    .chart {
        width: 100%;
        max-width: 760px;
        height: auto;
        margin: 12px auto;
    }

    .table-wrap { overflow-x: auto; }

    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        text-align: left;
    }

    th, td {
        padding: 12px 10px;
        border-bottom: 1px solid #ddd;
    }

    th {
        background: #e8f5e9;
        color: #217346;
    }

    .error { color: #b00020; }
    .note { font-size: 13px; color: #666; }

    @media (max-width: 500px) {
        .container { padding: 17px; }
        h1 { font-size: 25px; }
    }
</style>
"""


# -------------------- CLASSIFICATION PAGE --------------------

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Recycling Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {{ style|safe }}
</head>
<body>
<div class="container">

    <h1 style="text-align:center;">♻️ Smart Recycling Assistant</h1>

    <p style="text-align:center;">
        Identify waste using AI and learn how to recycle,
        compost, or dispose of it responsibly.
    </p>

    <div class="nav">
        <a class="button" href="/">Classify Waste</a>
        <a class="button" href="/dashboard">Recycling Dashboard</a>
    </div>

    <form method="POST" enctype="multipart/form-data"
          style="text-align:center;">
        <input type="file" name="image" accept="image/*" required>
        <br>
        <button class="button" type="submit">Identify Waste</button>
    </form>

    {% if category %}
    <div class="result">
        <h2>🔍 Waste Classification Result</h2>

        <img class="uploaded-image"
             src="data:image/jpeg;base64,{{ image_data }}"
             alt="Uploaded waste">

        <p><strong>Predicted category:</strong> {{ category|title }}</p>
        <p><strong>Model confidence:</strong> {{ confidence }}%</p>

        <h3>♻️ Recommended Treatment</h3>
        <div class="method">{{ guide.method }}</div>

        <h3>How to Handle This Waste</h3>
        <ol>
            {% for step in guide.steps %}
            <li>{{ step }}</li>
            {% endfor %}
        </ol>

        <h3>🌱 Decomposition Information</h3>
        <p>{{ guide.decomposition }}</p>

        <div class="tip">
            <strong>Helpful tip:</strong> {{ guide.tip }}
        </div>

        <p class="note">
            Recommendations are general guidance. Follow local
            waste-management rules. AI predictions may be incorrect.
        </p>
    </div>
    {% endif %}

    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}

</div>
</body>
</html>
"""


# -------------------- TENSORFLOW LITE PREDICTION --------------------

def predict_waste(image_array):
    """Run inference with the TensorFlow Lite model."""

    input_info = input_details[0]
    output_info = output_details[0]

    input_dtype = input_info["dtype"]

    # Handle floating-point and quantized integer model inputs.
    if np.issubdtype(input_dtype, np.integer):
        scale, zero_point = input_info["quantization"]

        if scale <= 0:
            raise ValueError("Invalid input quantization scale.")

        image_array = np.round(
            image_array / scale + zero_point
        )

        limits = np.iinfo(input_dtype)
        image_array = np.clip(
            image_array, limits.min, limits.max
        ).astype(input_dtype)
    else:
        image_array = image_array.astype(input_dtype)

    interpreter.set_tensor(
        input_info["index"], image_array
    )
    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_info["index"]
    )[0]

    # Dequantize integer output when necessary.
    if np.issubdtype(predictions.dtype, np.integer):
        scale, zero_point = output_info["quantization"]

        if scale > 0:
            predictions = (
                predictions.astype(np.float32) - zero_point
            ) * scale

    predictions = np.asarray(predictions, dtype=np.float32)

    if predictions.size != len(class_names):
        raise ValueError(
            "Model output does not match the six waste categories."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model returned invalid prediction values.")

    # This assumes the model output represents class probabilities.
    # If it outputs logits, the values need different interpretation.
    index = int(np.argmax(predictions))
    confidence = round(float(predictions[index]) * 100, 2)

    return class_names[index], confidence


# -------------------- HOME ROUTE --------------------

@app.route("/", methods=["GET", "POST"])
def home():
    category = None
    confidence = None
    image_data = None
    guide = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an image."
        else:
            try:
                image_bytes = uploaded_file.read()

                with Image.open(io.BytesIO(image_bytes)) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1200, 1200))

                    # Prepare image for display.
                    display_buffer = io.BytesIO()
                    img.save(display_buffer, format="JPEG")

                    image_data = base64.b64encode(
                        display_buffer.getvalue()
                    ).decode("utf-8")

                    # Prepare image for the model.
                    model_img = img.resize((128, 128))
                    image_array = np.asarray(
                        model_img, dtype=np.float32
                    )
                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                category, confidence = predict_waste(image_array)

                guide = waste_guide.get(
                    category.lower(), waste_guide["trash"]
                )

                save_prediction(category, confidence)

            except Exception as exc:
                print("Classification error:", exc)
                error = (
                    "Could not process this image. "
                    "Please select another image."
                )

    return render_template_string(
        HOME_HTML,
        style=STYLE,
        category=category,
        confidence=confidence,
        image_data=image_data,
        guide=guide,
        error=error
    )


# -------------------- DASHBOARD PAGE --------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Recycling Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {{ style|safe }}
</head>
<body>
<div class="container">

    <h1 style="text-align:center;">♻️ Recycling Dashboard</h1>

    <p style="text-align:center;">
        View your waste classification history and category distribution.
    </p>

    <div class="nav">
        <a class="button" href="/">Classify New Waste</a>
        <a class="button" href="/dashboard">Refresh Dashboard</a>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div>Total Classifications</div>
            <div class="stat-number">{{ total }}</div>
        </div>

        <div class="stat-card">
            <div>Categories Detected</div>
            <div class="stat-number">{{ category_total }}</div>
        </div>

        <div class="stat-card">
            <div>Recycling-Suitable Categories*</div>
            <div class="stat-number">{{ recyclable_categories }}</div>
        </div>
    </div>

    <div class="card">
        <h2>📊 Waste Distribution</h2>

        {% if chart_html %}
            <img class="chart"
                 src="data:image/png;base64,{{ chart_html }}"
                 alt="Waste classification distribution chart">
        {% else %}
            <p>No classifications yet. Classify an image first
               to create your dashboard history.</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>📋 Category Summary</h2>

        {% if categories %}
        <div class="table-wrap">
        <table>
            <tr>
                <th>Waste Category</th>
                <th>Number of Images</th>
            </tr>
            {% for item in categories %}
            <tr>
                <td>{{ item["category"]|title }}</td>
                <td>{{ item["count"] }}</td>
            </tr>
            {% endfor %}
        </table>
        </div>
        {% else %}
            <p>No category statistics available yet.</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>🕒 Recent Classifications</h2>

        {% if recent %}
        <div class="table-wrap">
        <table>
            <tr>
                <th>Category</th>
                <th>Confidence</th>
                <th>Date and Time</th>
            </tr>
            {% for item in recent %}
            <tr>
                <td>{{ item["category"]|title }}</td>
                <td>{{ "%.2f"|format(item["confidence"]) }}%</td>
                <td>{{ item["created_at"] }}</td>
            </tr>
            {% endfor %}
        </table>
        </div>
        {% else %}
            <p>No history available. Upload a waste image first.</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>🌱 Recycling Tips</h2>
        <ul>
            <li>Separate waste into appropriate categories.</li>
            <li>Keep recyclable materials clean and dry when required.</li>
            <li>Compost suitable food scraps and untreated paper materials.</li>
            <li>Never burn plastic or mix hazardous waste with household waste.</li>
            <li>Follow your local recycling collection rules.</li>
        </ul>
    </div>

    <p class="note">
        *The category count is based on categories classified so far,
        not a measurement of the actual quantity recycled.
        Classification confidence does not guarantee prediction accuracy.
    </p>

</div>
</body>
</html>
"""


@app.route("/dashboard")
def dashboard():
    total, categories, recent = get_dashboard_data()

    category_total = len(categories)

    recyclable_names = {
        "cardboard", "glass", "metal", "paper", "plastic"
    }

    recyclable_categories = sum(
        1 for item in categories
        if item["category"].lower() in recyclable_names
    )

    chart_html = None

    if categories:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        category_names = [
            item["category"].title() for item in categories
        ]
        category_counts = [
            item["count"] for item in categories
        ]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(category_names, category_counts)
        ax.set_title("Waste Classification Distribution")
        ax.set_xlabel("Waste Category")
        ax.set_ylabel("Number of Classifications")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)

        chart_html = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    return render_template_string(
        DASHBOARD_HTML,
        style=STYLE,
        total=total,
        categories=categories,
        recent=recent,
        category_total=category_total,
        recyclable_categories=recyclable_categories,
        chart_html=chart_html
    )


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    app.run(debug=True)
