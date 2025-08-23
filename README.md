# Kidaura Touch Data Analysis System

Advanced touch data analysis system with ML enhancement, data validation, and interactive visualizations for Kidaura research platform.

## 🚀 Features

- **Complete Data Processing Pipeline**: Automated processing of touch data from JSON to CSV with comprehensive validation
- **ML-Enhanced Analysis**: Advanced machine learning pipeline with feature engineering and quality assessment
- **Data Validation & Flagging**: Sophisticated validation system with Touchdata_id-based sequence validation
- **Google Sheets Integration**: Seamless export to Google Sheets with automatic sharing capabilities
- **Interactive Visualizations**: HTML-based interactive visualizations and comprehensive documentation
- **Unified Launcher System**: Easy-to-use command-line interface for all system components

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── core/              # Data processing engine
│   ├── ml/                # Machine learning components
│   ├── export/            # Google Sheets export functionality
│   ├── utils/             # Utility functions
│   └── visualization/     # Interactive visualization tools
├── ML/                    # Standalone ML pipeline
├── data/                  # Data storage (raw, processed, outputs)
├── tests/                 # Comprehensive test suite
├── docs/                  # Documentation and guides
├── config/                # Configuration files
└── scripts/               # Utility scripts
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jbanmol/Kidaura-Touchdata-Analysis.git
   cd Kidaura-Touchdata-Analysis
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google Sheets** (optional):
   - Place your Google Sheets credentials in `config/credentials.json`
   - Follow the setup guide in `docs/user_guide/`

## 🚀 Quick Start

### Using the Unified Launcher
```bash
python app.py
```

### Direct Component Access
```bash
# Data processing
python src/core/data_processor.py

# ML enhancement
python src/ml/consolidated_enhancer.py

# Google Sheets export
python src/export/google_sheets.py

# Interactive visualizations
python src/visualization/html_interactive.py
```

## 📊 Data Processing Pipeline

1. **Raw Data Input**: JSON/CSV touch data files
2. **Validation & Flagging**: Sequence validation with Touchdata_id support
3. **ML Enhancement**: Feature engineering and quality assessment
4. **Export Options**: CSV, Google Sheets, interactive HTML
5. **Visualization**: Interactive charts and data exploration tools

## 🧠 Machine Learning Features

- **Quality Score**: 0-100 weighted metric for data quality assessment
- **Interaction Type**: Classification (Precise/Quick/Hesitant/Erratic)
- **Anomaly Detection**: Technical, behavioral, and spatial anomaly flagging
- **Research Suitability**: Automated tagging for different analysis types

## 📈 Visualization & Documentation

- **Interactive HTML Reports**: Comprehensive data exploration interface
- **Real-time Statistics**: Dynamic data quality metrics
- **Visual Flow Diagrams**: System architecture and data flow visualization
- **Mobile-Responsive Design**: Access from any device

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python tests/test_integration.py
python tests/test_ml_integration.py
python tests/performance_test.py
```

## 📚 Documentation

- **User Guide**: `docs/user_guide/`
- **Developer Guide**: `docs/developer_guide/`
- **API Documentation**: `docs/about_app.md`
- **Interactive Documentation**: Open `about.html` in your browser

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 👥 Authors

- **jbanmol** - *Initial work and development*

## 🙏 Acknowledgments

- Kidaura research team for requirements and testing
- Contributors to the touch data analysis methodology
- Open source libraries that made this project possible

---

For detailed usage instructions, see the [Unified Launcher Guide](UNIFIED_LAUNCHER_GUIDE.md).
