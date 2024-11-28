import os
import shutil
from datetime import datetime
from jinja2 import Template
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from config.config import config

temporary_output_folder = config["temporary_output_folder"]


class ReportGenerator:
    def __init__(self, output_dir=temporary_output_folder):
        self.output_dir = output_dir
        self.temp_dir = "temporary_output"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(output_dir, f"report_{self.timestamp}")
        self.assets_dir = os.path.join(self.report_dir, "assets")
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for the report"""
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)

    def _copy_temp_files(self):
        """Copy all relevant files from temporary_output to report assets"""
        # Copy all image files
        for filename in os.listdir(self.temp_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(self.temp_dir, filename)
                dst = os.path.join(self.assets_dir, filename)
                shutil.copy2(src, dst)

    def generate_report(self, sampleholder, optimization_results=None):
        """Generate HTML report with all results"""
        self._copy_temp_files()

        # Collect all images in assets directory
        image_files = sorted(
            [
                f
                for f in os.listdir(self.assets_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        # Add parameters to optimization_results if they exist
        if optimization_results is None:
            optimization_results = {}

        # Add contour finding parameters to sampleholder
        if sampleholder and not hasattr(sampleholder, "contour_finding_params"):
            from config.config import image2contours_kwargs

            sampleholder.contour_finding_params = image2contours_kwargs

        # Add optimization parameters if they don't exist
        if optimization_results and not hasattr(optimization_results, "parameters"):
            from config.config import batch_optimization_kwargs

            optimization_results.update(batch_optimization_kwargs)

        # Prepare data for the template
        template_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_files": image_files,
            "sampleholder": sampleholder,
            "optimization_results": optimization_results,
        }

        # Generate HTML using template
        html_content = self._generate_html(template_data)

        # Write HTML file
        report_path = os.path.join(self.report_dir, "report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Copy CSS file
        self._copy_css_file()

        return report_path

    def _generate_html(self, data):
        """Generate HTML content using template"""
        template = Template(
            """
<!DOCTYPE html>
<html>
<head>
    <title>Processing Report - {{ timestamp }}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Image Processing and Optimization Report</h1>
        <p class="timestamp">Generated on: {{ timestamp }}</p>
        
        <section class="parameters">
            <h2>Processing Parameters</h2>
            
            <div class="parameters-grid">
                <div class="parameter-section">
                    <h3>Contour Finding Parameters</h3>
                    <div class="parameter-content">
                        <p><strong>Epsilon:</strong> {{ "%.3f"|format(sampleholder.contour_finding_params.epsilon) if sampleholder.contour_finding_params.epsilon is not none else "N/A" }}</p>
                        <p><strong>Lowercut:</strong> {{ sampleholder.contour_finding_params.lowercut if sampleholder.contour_finding_params.lowercut is not none else "N/A" }}</p>
                        <p><strong>Area Lowercut:</strong> {{ sampleholder.contour_finding_params.area_lowercut if sampleholder.contour_finding_params.area_lowercut is not none else "N/A" }}</p>
                        <p><strong>Threshold:</strong> {{ sampleholder.contour_finding_params.threshold if sampleholder.contour_finding_params.threshold is not none else "N/A" }}</p>
                        <p><strong>Gaussian Filter:</strong> {{ "Enabled" if sampleholder.contour_finding_params.is_gaussian_filter else "Disabled" }}</p>
                        {% if sampleholder.contour_finding_params.is_gaussian_filter %}
                        <p><strong>Gaussian Window:</strong> {{ sampleholder.contour_finding_params.gaussian_window|join(', ') }}</p>
                        {% endif %}
                    </div>
                </div>

                <div class="parameter-section">
                    <h3>Close Packing Parameters</h3>
                    <div class="parameter-content">
                        <p><strong>Number of Systems:</strong> {{ optimization_results.number_system if optimization_results.number_system is not none else "N/A" }}</p>
                        <p><strong>Step Size:</strong> {{ "%.2f"|format(optimization_results.step_size) if optimization_results.step_size is not none else "N/A" }}</p>
                        <p><strong>Number of Iterations:</strong> {{ optimization_results.iterations if optimization_results.iterations is not none else "N/A" }}</p>
                        <p><strong>Temperature:</strong> {{ "%.2f"|format(optimization_results.temperature) if optimization_results.temperature is not none else "N/A" }}</p>
                        <p><strong>Gravity Multiplier:</strong> {{ "%.2f"|format(optimization_results.gravity_multiplier) if optimization_results.gravity_multiplier is not none else "N/A" }}</p>
                        <p><strong>Gravity Off At:</strong> {{ optimization_results.gravity_off_at if optimization_results.gravity_off_at is not none else "N/A" }}</p>
                        <p><strong>Contour Buffer Multiplier:</strong> {{ "%.2f"|format(optimization_results.contour_buffer_multiplier) if optimization_results.contour_buffer_multiplier is not none else "N/A" }}</p>
                        <p><strong>Optimize Shape:</strong> {{ optimization_results.optimize_shape if optimization_results.optimize_shape is not none else "N/A" }}</p>
                    </div>
                </div>
            </div>
        </section>

        <section class="summary">
            <h2>Summary</h2>
            <div class="summary-content">
                <p>Number of samples: {{ sampleholder.number_samples() }}</p>
                <p>Total samples area: {{ "%.2f" % sampleholder.samples_area }}</p>
                <p>Packing ratio: {{ "%.2f" % (sampleholder.ratio * 100) }}%</p>
            </div>
        </section>

        <section class="results">
            <h2>Processing Results</h2>
            <div class="image-gallery">
            {% for image in image_files %}
                <div class="image-container">
                    <img src="assets/{{ image }}" alt="{{ image }}">
                    <p class="image-caption">{{ image }}</p>
                </div>
            {% endfor %}
            </div>
        </section>
        
        {% if optimization_results %}
        <section class="optimization">
            <h2>Optimization Results</h2>
            <div class="optimization-content">
                <p>Final area ratio: {{ "%.2f" % (optimization_results.final_ratio * 100) }}%</p>
                <p>Number of iterations: {{ optimization_results.iterations }}</p>
                <p>Computation time: {{ "%.2f" % optimization_results.computation_time }}s</p>
            </div>
        </section>
        {% endif %}
    </div>
</body>
</html>
        """
        )

        return template.render(**data)

    def _copy_css_file(self):
        """Create and copy CSS file to report directory"""
        css_content = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    text-align: center;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 10px;
}

h2 {
    color: #4CAF50;
    margin-top: 30px;
}

.timestamp {
    text-align: right;
    color: #666;
    font-style: italic;
}

.summary-content {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}

.image-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.image-container {
    text-align: center;
}

.image-container img {
    max-width: 100%;
    height: auto;
    border-radius: 5px;
    box-shadow: 0 0 5px rgba(0,0,0,0.2);
}

.image-caption {
    margin-top: 10px;
    color: #666;
    font-size: 0.9em;
}

.optimization-content {
    background-color: #f0f7f0;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}

section {
    margin-bottom: 40px;
}

/* Parameters Section Styling */
.parameters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.parameter-section {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.parameter-section h3 {
    color: #2196F3;
    margin-top: 0;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e0e0e0;
}

.parameter-content {
    font-size: 14px;
}

.parameter-content p {
    margin: 8px 0;
    line-height: 1.4;
}

.parameter-content strong {
    color: #555;
    font-weight: 600;
}
        """

        css_path = os.path.join(self.report_dir, "style.css")
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(css_content)
