# 🌳 Drubo Live — Urban Forest Benchmark Platform

## ঢাকার নগর বনের গঠন ও কার্বন ভাণ্ডার বেঞ্চমার্ক প্ল্যাটফর্ম

An interactive web platform where scientists upload their field survey data (CSV) and satellite rasters (GeoTIFF), and receive instant benchmark grading results with interactive maps and charts.

---

## 🎯 Features / বৈশিষ্ট্য

| Feature | Description |
|---------|-------------|
| 📊 Full Benchmark Mode | Upload survey CSV → spatial block split → raster extraction → grading |
| 📝 Quick Grade Mode | Upload prediction + ground truth → instant score |
| 🗺️ Interactive Maps | Plotly-based scatter maps showing train/test split on Dhaka |
| 📈 Score Gauges | Visual gauge charts for per-target and overall scores |
| 🔍 Scatter Plots | Predicted vs Actual scatter for each of 5 targets |
| ⬇️ Downloadable Reports | JSON and CSV exports of grading results |
| 🌐 Bengali + English | Bilingual interface for local researchers |

---

## 🚀 Quick Start / দ্রুত শুরু

### Local Run / স্থানীয়ভাবে চালান

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open in browser
# http://localhost:8501
```

### Hugging Face Spaces Deployment / হাগিং ফেস-এ ডিপ্লয়

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as SDK
3. Upload all files from this `Drubo_live/` folder
4. The app will auto-deploy (free tier supported)

---

## 📁 Folder Structure / ফোল্ডার কাঠামো

```
Drubo_live/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── grading_engine.py       # Scoring logic (RMSE, R², baseline)
│   ├── raster_processor.py     # GeoTIFF reading & pixel sampling
│   └── spatial_split.py        # Spatial block train/test split
├── uploads/                    # User-uploaded files (auto-created)
│   ├── survey/
│   ├── rasters/
│   └── predictions/
├── results/                    # Output reports (auto-created)
└── assets/                     # Static assets (logos, etc.)
```

---

## 📊 Input Data Format / ইনপুট ডেটা ফরম্যাট

### Field Survey CSV

```csv
plot_id,lat,long,stratum,land_use,stem_count,basal_area,carbon_storage,leaf_area,percent_tree_cover
DK_PL_0001,23.767,90.410,Zone_1_Old_Central,Residential,5,0.234,1.567,45.6,38.5
...
```

**Required columns:**
- `plot_id` — unique identifier
- `lat`, `long` — WGS84 coordinates
- `stratum` — survey stratum (e.g., Zone_1, Zone_2, Zone_3)
- `land_use` — land use class (Residential, Commercial, etc.)
- 5 target columns: `stem_count`, `basal_area`, `carbon_storage`, `leaf_area`, `percent_tree_cover`

### Satellite Rasters (GeoTIFF)

| File | Description |
|------|-------------|
| `sentinel2_b4_red.tif` | Sentinel-2 Band 4 (Red, ~665nm) |
| `sentinel2_b8_nir.tif` | Sentinel-2 Band 8 (NIR, ~842nm) |
| `sentinel1_vh.tif` | Sentinel-1 VH polarization backscatter |
| `canopy_height.tif` | Canopy Height Model (CHM) |

### Prediction CSV

```csv
plot_id,stem_count,basal_area,carbon_storage,leaf_area,percent_tree_cover
DK_PL_0002,3,0.189,0.956,34.2,25.7
...
```

---

## 📐 Scoring Formula / স্কোরিং সূত্র

```
Score_t = clamp(1 − RMSE_agent / RMSE_baseline, 0, 1)
Final Score = mean(Score_t for all 5 targets)
```

- **Baseline**: Stratum × Land-use group mean (naïve predictor computed from training set)
- **Range**: 0.0 (worse than baseline) to 1.0 (perfect prediction)
- **Gate 1 Threshold**: Mean score < 0.67 means the task is hard enough for AI agents

---

## 📚 Reference / তথ্যসূত্র

Majumder, S.C., et al. (2026). *Urban forests baseline and ecosystem benefits of a tropical
metropolis: case of Dhaka, Bangladesh.* Scientific Reports, 16, Article 46985.
DOI: [10.1038/s41598-026-46985-x](https://doi.org/10.1038/s41598-026-46985-x)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Charts**: Plotly
- **Raster I/O**: Pillow (fallback) / rasterio (optional)
- **Hosting**: Hugging Face Spaces (free) or any Python server

---

*Built by Drubo with Antigravity 🚀*
